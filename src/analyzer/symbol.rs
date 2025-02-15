use super::common::{AstFnDef, TypeAliasStmt, VarDeclExpr};
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};

// 定义索引类型
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct NodeId(NonZeroU32);

// 全局 scope id
pub const GLOBAL_SCOPE_ID: NodeId = unsafe { NodeId(NonZeroU32::new_unchecked(1)) };

// Arena 分配器
#[derive(Debug, Clone)]
pub struct Arena<T> {
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

//  引用自 AstNode
#[derive(Debug, Clone)]
pub enum SymbolKind {
    Var(Arc<Mutex<VarDeclExpr>>), // 变量原始定义
    Fn(Arc<Mutex<AstFnDef>>),
    TypeAlias(Arc<Mutex<TypeAliasStmt>>),
}

// symbol table 可以同时处理多个文件的 scope, 存在一个 global scope 管理所有的全局 scope, 符号注册到 global scope 时，define_ident 需要携带 package_name 保证符号的唯一性
#[derive(Debug, Clone)]
pub struct Symbol {
    // local symbol 直接使用，global symbol 会携带 package ident
    pub ident: String,
    pub kind: SymbolKind,
    pub defined_in: NodeId, // defined in scope
    pub pos: usize,         // 符号定义的其实位置

    // local symbol 需要一些额外信息
    pub is_capture: bool, // 如果变量被捕获，则需要分配到堆中，避免作用域问题

    pub generics_id_map: HashMap<String, NodeId>, // new ident -> new symbol id
}

#[derive(Debug, Clone)]
pub struct FreeIdent {
    pub in_parent_local: bool,   // 是否在父作用域中直接定义, 否则需要通过 envs[index] 访问
    pub parent_env_index: usize, // 父作用域起传递作用, 通过 envs 参数向下传递

    pub index: usize, // free in frees index
    pub ident: String,
    pub kind: SymbolKind,
}

#[derive(Debug, Clone)]
pub enum ScopeKind {
    Global,
    GlobalFn(Arc<Mutex<AstFnDef>>),
    LocalFn(Arc<Mutex<AstFnDef>>),
    Local,
}

#[derive(Debug, Clone)]
pub struct Scope {
    pub parent: Option<NodeId>,              // 除了全局作用域外，每个作用域都有一个父作用域
    pub symbols: Vec<NodeId>,                // 当前作用域中定义的符号列表
    pub children: Vec<NodeId>,               // 子作用域列表
    pub symbol_map: HashMap<String, NodeId>, // 符号名到符号ID的映射

    pub range: (usize, usize), // 作用域的范围, [start, end)

    // 当前作用域是否为函数级别作用域
    pub kind: ScopeKind,

    pub frees: HashMap<String, FreeIdent>, // fn scope 需要处理函数外的自由变量
}

#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub scopes: Arena<Scope>,     // 作用域列表, 根据 NodeId 索引
    pub symbols: Arena<Symbol>,   // 符号列表, 根据 NodeId 索引
    pub current_scope_id: NodeId, // ast 解析时记录当前作用域 id
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

    pub fn get_scope(&self) -> &Scope {
        self.scopes.get(self.current_scope_id).unwrap()
    }

    pub fn find_scope(&self, scope_id: NodeId) -> Option<&Scope> {
        self.scopes.get(scope_id)
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

    /**
     * 存在 rebuild 机制，所以同一个符号会被重复定义, 但是定义的位置相同。
     */
    pub fn define_symbol(&mut self, ident: String, kind: SymbolKind, pos: usize) -> Result<NodeId, String> {
        // 检查当前作用域是否已存在同名符号
        if let Some(scope) = self.scopes.get(self.current_scope_id) {
            if let Some(&existing_symbol_id) = scope.symbol_map.get(&ident) {
                // 获取已存在的符号
                if let Some(existing_symbol) = self.symbols.get_mut(existing_symbol_id) {
                    // 如果位置相同，则更新 kind
                    if existing_symbol.pos == pos {
                        existing_symbol.kind = kind;
                        return Ok(existing_symbol_id);
                    } else {
                        // 位置不同，则是真正的重复定义
                        return Err(format!(
                            "symbol '{}' already defined in current scope at position {}",
                            ident, existing_symbol.pos
                        ));
                    }
                }
            }
        }

        let symbol = Symbol {
            ident: ident.clone(),
            kind,
            defined_in: self.current_scope_id,
            pos,
            is_capture: false,
            generics_id_map: HashMap::new(),
        };

        let symbol_id = self.symbols.alloc(symbol);

        // 将符号添加到当前作用域
        if let Some(scope) = self.scopes.get_mut(self.current_scope_id) {
            scope.symbols.push(symbol_id);
            scope.symbol_map.insert(ident, symbol_id);
        }

        Ok(symbol_id)
    }

    // 查找符号（包括父作用域）
    pub fn lookup_symbol(&self, ident: &str) -> Option<NodeId> {
        let mut current = Some(self.current_scope_id);

        while let Some(scope_id) = current {
            if let Some(scope) = self.scopes.get(scope_id) {
                // 在当前作用域中查找
                if let Some(&symbol_id) = scope.symbol_map.get(ident) {
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

    pub fn find_global_fn(&self) -> Option<Arc<Mutex<AstFnDef>>> {
        let mut current = Some(self.current_scope_id);

        while let Some(scope_id) = current {
            if let Some(scope) = self.scopes.get(scope_id) {
                match &scope.kind {
                    ScopeKind::GlobalFn(fn_def) => return Some(fn_def.clone()),
                    _ => current = scope.parent,
                }
            } else {
                break;
            }
        }
        None
    }

    pub fn find_symbol_id(&self, ident: &str, scope_id: NodeId) -> Option<NodeId> {
        if let Some(scope) = self.scopes.get(scope_id) {
            return scope.symbol_map.get(ident).cloned();
        } else {
            return None;
        }
    }

    pub fn find_global_symbol(&self, ident: &str) -> Option<&Symbol> {
        if let Some(symbol_id) = self.find_symbol_id(ident, GLOBAL_SCOPE_ID) {
            return Some(self.symbols.get(symbol_id).unwrap());
        } else {
            return None;
        }
    }

    pub fn symbol_exists_in_scope(&self, ident: &str, scope_id: NodeId) -> bool {
        if let Some(scope) = self.scopes.get(scope_id) {
            return scope.symbol_map.contains_key(ident);
        } else {
            return false;
        }
    }

    pub fn symbol_exists_in_current(&self, ident: &str) -> bool {
        if let Some(scope) = self.scopes.get(self.current_scope_id) {
            return scope.symbol_map.contains_key(ident);
        } else {
            return false;
        }
    }

    pub fn get_symbol(&mut self, id: NodeId) -> Option<&mut Symbol> {
        self.symbols.get_mut(id)
    }

    // 打印作用域树（用于调试）
    pub fn print_scope_tree(&self, scope_id: NodeId, indent: usize) {
        if let Some(scope) = self.scopes.get(scope_id) {
            println!("{}Scope {:?}:", " ".repeat(indent), scope_id);

            // 打印该作用域中的符号
            for &symbol_id in &scope.symbols {
                if let Some(symbol) = self.symbols.get(symbol_id) {
                    println!("{}  Symbol: {} ({:?})", " ".repeat(indent), symbol.ident, symbol.kind);
                }
            }

            // 递归打印子作用域
            for &child_id in &scope.children {
                self.print_scope_tree(child_id, indent + 2);
            }
        }
    }
}
