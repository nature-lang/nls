use super::common::*;
use super::symbol::{ScopeKind, SymbolKind, SymbolTable};
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};

pub struct SemanticError(usize, usize, String);

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SyntaxError: {}", self.2)
    }
}

impl fmt::Debug for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SemanticError: {}", self.2)
    }
}
impl Error for SemanticError {}

pub struct Semantic {
    symbol_table: SymbolTable,
    errors: Vec<AnalyzerError>,
    package_name: String,
    stmts: Vec<Box<Stmt>>,
    fn_list: Vec<Arc<Mutex<AstFnDef>>>,
}

impl Semantic {
    pub fn new(package_name: String, stmts: Vec<Box<Stmt>>) -> Self {
        Self {
            symbol_table: SymbolTable::new(),
            errors: Vec::new(),
            package_name,
            stmts,
            fn_list: Vec::new(),
        }
    }

    fn analyze_type(&mut self, type_: &Type) {
        todo!();
    }

    pub fn analyze(&mut self) -> (Vec<Box<Stmt>>, Vec<AnalyzerError>) {
        let mut fn_stmt_list = Vec::<Arc<Mutex<AstFnDef>>>::new();

        let mut var_assign_list = Vec::<Box<Stmt>>::new();

        let mut stmt_list = Vec::<Box<Stmt>>::new();

        // 跳过 import
        for i in 0..self.stmts.len() {
            // 使用 clone 避免对 self 所有权占用
            let stmt = self.stmts[i].clone();

            match &stmt.node {
                AstNode::Import(..) => continue,
                AstNode::FnDef(fndef_mutex) => {
                    let mut fndef = fndef_mutex.lock().unwrap();
                    let mut symbol_name = fndef.symbol_name.clone();

                    // fn string<T>.len() -> fn <T>.string_len to symbol_table
                    if !fndef.impl_type.kind.is_unknown() {
                        assert!(fndef.impl_type.impl_ident != None);
                        symbol_name = format!("{}_{}", fndef.impl_type.impl_ident.as_ref().unwrap(), symbol_name)
                    }

                    // fndef 的 symbol_name 需要包含 package name 来构成全局搜索名称
                    // package_name.symbol_name
                    fndef.symbol_name = format!("{}.{}", self.package_name, symbol_name);

                    if let Err(e) = self.symbol_table.define_symbol(
                        fndef.symbol_name.clone(),
                        fndef.symbol_name.clone(),
                        SymbolKind::Fn(fndef_mutex.clone()),
                        fndef.symbol_start,
                    ) {
                        self.errors.push(AnalyzerError {
                            start: fndef.symbol_start,
                            end: fndef.symbol_end,
                            message: e,
                        });
                    }

                    fn_stmt_list.push(fndef_mutex.clone());
                }
                AstNode::VarDef(var_decl_mutex, expr) => {
                    let mut var_decl = var_decl_mutex.lock().unwrap();


                    // ident rewrite
                    var_decl.ident = format!("{}.{}", self.package_name, var_decl.ident);

                    // 添加到符号表
                    if let Err(e) = self.symbol_table.define_symbol(
                        var_decl.ident.clone(),
                        var_decl.ident.clone(),
                        SymbolKind::Var(var_decl_mutex.clone()),
                        var_decl.symbol_start,
                    ) {
                        self.errors.push(AnalyzerError {
                            start: var_decl.symbol_start,
                            end: var_decl.symbol_end,
                            message: e,
                        });
                    }

                    self.analyze_type(&var_decl.type_);

                    // 将 vardef 转换成 assign 导入到 package init 中进行初始化
                    let assign_left = Box::new(Expr::ident(
                        var_decl.symbol_start,
                        var_decl.symbol_end,
                        var_decl.ident.clone(),
                    ));
                    let assign_stmt = Box::new(Stmt {
                        node: AstNode::Assign(assign_left, expr.clone()),
                        start: expr.start,
                        end: expr.end,
                    });
                    var_assign_list.push(assign_stmt);
                }
                AstNode::TypeAlias(type_alias_mutex) => {
                    let mut type_alias = type_alias_mutex.lock().unwrap();

                    type_alias.ident = format!("{}.{}", self.package_name, type_alias.ident);
                    // 添加到符号表
                    if let Err(e) = self.symbol_table.define_symbol(
                        type_alias.ident.clone(),
                        type_alias.ident.clone(),
                        SymbolKind::TypeAlias(type_alias_mutex.clone()),
                        type_alias.symbol_start,
                    ) {
                        self.errors.push(AnalyzerError {
                            start: type_alias.symbol_start,
                            end: type_alias.symbol_end,
                            message: e,
                        });
                    }

                    // 处理 params constraints, type foo<T:int|float, E:int:bool> = ...
                    if let Some(ref params) = type_alias.params {
                        for param in params {
                            if !param.constraints.0 {
                                // 遍历所有 constraints 类型 进行 analyze
                                for constraint in &param.constraints.1 {
                                    self.analyze_type(&constraint);
                                }
                            }
                        }
                    }
                }
                _ => {
                    // 语义分析中包含许多错误
                }
            }

            // 归还 stmt list
            stmt_list.push(stmt);
        }

        // 封装 fn init
        if !var_assign_list.is_empty() {
            // 创建init函数定义
            let mut fn_init = AstFnDef::default();
            fn_init.symbol_name = format!("{}.{}", self.package_name, "init");
            fn_init.fn_name = Some(fn_init.symbol_name.clone());
            fn_init.return_type = Type::new(TypeKind::Void);
            fn_init.body = var_assign_list;

            fn_stmt_list.push(Arc::new(Mutex::new(fn_init)));
        }

        // 对 fn stmt list 进行 analyzer 处理。
        for fndef_mutex in fn_stmt_list {
            let fndef = fndef_mutex;
            self.analyze_global_fn(fndef.clone());
        }

        (stmt_list, self.errors.clone())
    }

    pub fn analyze_global_fn(&mut self, fndef_mutex: Arc<Mutex<AstFnDef>>) -> Result<(), SemanticError> {
        let mut fndef = fndef_mutex.lock().unwrap();
        fndef.is_local = false;
        if fndef.generics_params.is_some() {
            fndef.is_generics = true;
        }

        if fndef.is_tpl {
            assert!(fndef.body.len() == 0);
            if fndef.impl_type.kind.is_exist() {
                return Err(SemanticError(
                    fndef.start,
                    fndef.end,
                    "tpl fn cannot have impl type".to_string(),
                ));
            }
        }

        self.analyze_type(&fndef.return_type);

        self.symbol_table.enter_create_scope(ScopeKind::Fn);

        if fndef.impl_type.kind.is_exist() {
            // 如果 impl type 是 type alias, 则从符号表中获取当前的 type alias 的全称进行更新
            todo!();
        }

        // 处理 params
        

        Ok(())
    }
}
