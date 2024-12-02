use super::common::{AstFnDef, VarDeclExpr, TypeAliasStmt};
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};

// 定义索引类型
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct NodeId(NonZeroU32);

// 全局 scope id
const GLOBAL_SCOPE_ID: NodeId = unsafe { NodeId(NonZeroU32::new_unchecked(1)) };

// Arena 分配器
#[derive(Debug)]
struct Arena<T> {
    items: Vec<T>,
}

impl<T> Arena<T> {
    fn new() -> Self {
        Arena { items: Vec::new() }
    }

    fn alloc(&mut self, value: T) -> NodeId {
        let id = self.items.len();
        self.items.push(value);
        // 确保 id + 1 不为 0
        NodeId(NonZeroU32::new(id as u32 + 1).unwrap())
    }

    fn get(&self, id: NodeId) -> Option<&T> {
        self.items.get(id.0.get() as usize - 1)
    }

    fn get_mut(&mut self, id: NodeId) -> Option<&mut T> {
        self.items.get_mut(id.0.get() as usize - 1)
    }
}

#[derive(Debug)]
pub enum SymbolKind {
    Var(Arc<Mutex<VarDeclExpr>>), // 变量原始定义
    Fn(Arc<Mutex<AstFnDef>>),
    TypeAlias(Arc<Mutex<TypeAliasStmt>>),
}

// symbol table 可以同时处理多个文件的 scope, 存在一个 global scope 管理所有的全局 scope, 符号注册到 global scope 时，define_ident 需要携带 package_name 保证符号的唯一性
#[derive(Debug)]
pub struct Symbol {
    // 符号定义的原始值, 用于符号的查找, 不同 scope 下可以有相同的 define_ident
    pub define_ident: String,
    pub kind: SymbolKind,
    pub defined_in: NodeId,
    pub pos: usize, // 符号定义的其实位置

    // local symbol 需要一些额外信息
    pub unique_ident: String, // 全局唯一标识, 用于在全局符号表中快速定位符号信息
    pub is_capture: bool,     // 如果变量被捕获，则需要分配到堆中，避免作用域问题
}

#[derive(Debug)]
pub struct FreeIdent {
    in_parent_local: bool,   // 是否在父作用域中直接定义, 否则需要通过 envs[index] 访问
    parent_env_index: usize, // 父作用域起传递作用, 通过 envs 参数向下传递

    index: usize, // free in frees index
    ident: String,
}

#[derive(Debug)]
pub enum ScopeKind {
    Global,
    Fn,
    Local,
}

#[derive(Debug)]
struct Scope {
    parent: Option<NodeId>,              // 除了全局作用域外，每个作用域都有一个父作用域
    symbols: Vec<NodeId>,                // 当前作用域中定义的符号列表
    children: Vec<NodeId>,               // 子作用域列表
    symbol_map: HashMap<String, NodeId>, // 符号名到符号ID的映射
    range: (usize, usize),               // 作用域的范围, [start, end)

    // 当前作用域是否为函数级别作用域
    kind: ScopeKind,

    frees: HashMap<String, FreeIdent>, // fn scope 需要处理函数外的自由变量
}

#[derive(Debug)]
pub struct SymbolTable {
    scopes: Arena<Scope>,     // 作用域列表, 根据 NodeId 索引
    symbols: Arena<Symbol>,   // 符号列表, 根据 NodeId 索引
    current_scope_id: NodeId, // ast 解析时记录当前作用域 id
}

impl SymbolTable {
    pub fn new() -> Self {
        let mut scopes = Arena::new();

        // 创建全局作用域
        let global_scope = Scope {
            parent: None,
            symbols: Vec::new(),
            children: Vec::new(),
            symbol_map: HashMap::new(),
            range: (0, 0),
            kind: ScopeKind::Global,
            frees: HashMap::new(),
        };

        let global_scope_id = scopes.alloc(global_scope);
        assert_eq!(global_scope_id, GLOBAL_SCOPE_ID);

        SymbolTable {
            scopes,
            symbols: Arena::new(),
            current_scope_id: global_scope_id,
        }
    }

    // 创建新的作用域
    fn create_scope(&mut self, kind: ScopeKind) -> NodeId {
        let new_scope = Scope {
            parent: Some(self.current_scope_id),
            symbols: Vec::new(),
            children: Vec::new(),
            symbol_map: HashMap::new(),
            range: (0, 0),
            kind,
            frees: HashMap::new(),
        };

        let new_scope_id = self.scopes.alloc(new_scope);

        // 将新作用域添加到父作用域的children中, current cope 作为 parent
        if let Some(current_scope) = self.scopes.get_mut(self.current_scope_id) {
            current_scope.children.push(new_scope_id);
        }

        new_scope_id
    }

    pub fn enter_scope(&mut self, scope_id: NodeId) {
        self.current_scope_id = scope_id;
    }

    pub fn enter_create_scope(&mut self, kind: ScopeKind) -> NodeId {
        let scope_id = self.create_scope(kind);
        self.current_scope_id = scope_id;
        scope_id
    }

    // 退出当前作用域
    pub fn exit_scope(&mut self) {
        if let Some(scope) = self.scopes.get(self.current_scope_id) {
            if let Some(parent_id) = scope.parent {
                self.current_scope_id = parent_id;
            }
        }
    }

    // 在当前作用域中定义符号
    pub fn define_symbol(
        &mut self,
        unique_ident: String,
        define_ident: String,
        kind: SymbolKind,
        pos: usize,
    ) -> Result<NodeId, String> {
        // 检查当前作用域是否已存在同名符号
        if let Some(scope) = self.scopes.get(self.current_scope_id) {
            if scope.symbol_map.contains_key(&define_ident) {
                return Err(format!("Symbol '{}' already defined in current scope", define_ident));
            }
        }

        let symbol = Symbol {
            define_ident: define_ident.clone(),
            kind,
            defined_in: self.current_scope_id,
            pos,
            unique_ident: unique_ident.clone(),
            is_capture: false,
        };

        let symbol_id = self.symbols.alloc(symbol);

        // 将符号添加到当前作用域
        if let Some(scope) = self.scopes.get_mut(self.current_scope_id) {
            scope.symbols.push(symbol_id);
            scope.symbol_map.insert(define_ident, symbol_id);
        }

        Ok(symbol_id)
    }

    // 查找符号（包括父作用域）
    pub fn lookup_symbol(&self, name: &str) -> Option<NodeId> {
        let mut current = Some(self.current_scope_id);

        while let Some(scope_id) = current {
            if let Some(scope) = self.scopes.get(scope_id) {
                // 在当前作用域中查找
                if let Some(&symbol_id) = scope.symbol_map.get(name) {
                    return Some(symbol_id);
                }
                // 继续查找父作用域
                current = scope.parent;
            } else {
                break;
            }
        }
        None
    }

    // 获取符号信息
    pub fn get_symbol(&self, id: NodeId) -> Option<&Symbol> {
        self.symbols.get(id)
    }

    // 打印作用域树（用于调试）
    pub fn print_scope_tree(&self, scope_id: NodeId, indent: usize) {
        if let Some(scope) = self.scopes.get(scope_id) {
            println!("{}Scope {:?}:", " ".repeat(indent), scope_id);

            // 打印该作用域中的符号
            for &symbol_id in &scope.symbols {
                if let Some(symbol) = self.symbols.get(symbol_id) {
                    println!(
                        "{}  Symbol: {} ({:?})",
                        " ".repeat(indent),
                        symbol.unique_ident,
                        symbol.kind
                    );
                }
            }

            // 递归打印子作用域
            for &child_id in &scope.children {
                self.print_scope_tree(child_id, indent + 2);
            }
        }
    }
}
