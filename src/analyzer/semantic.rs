use super::common::*;
use super::symbol::{NodeId, Scope, ScopeKind, SymbolKind, SymbolTable, GLOBAL_SCOPE_ID};
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

#[derive(Debug)]
pub struct Semantic {
    symbol_table: SymbolTable,
    errors: Vec<AnalyzerError>,
    package_name: String,
    stmts: Vec<Box<Stmt>>,
    imports: Vec<Box<ImportStmt>>,
    current_local_fn_list: Vec<Arc<Mutex<AstFnDef>>>,
}

impl Semantic {
    pub fn new(package_name: String, stmts: Vec<Box<Stmt>>) -> Self {
        Self {
            symbol_table: SymbolTable::new(),
            errors: Vec::new(),
            package_name,
            stmts,
            current_local_fn_list: Vec::new(),
            imports: Vec::new(),
        }
    }

    fn analyze_special_type_rewrite(&mut self, type_: &mut Type) -> bool {
        assert!(matches!(type_.kind, TypeKind::Alias(..)));
        let TypeKind::Alias(type_alias) = type_.kind.clone() else { unreachable!() };

        // void ptr rewrite
        if type_alias.ident == TypeKind::VoidPtr.to_string() {
            type_.kind = TypeKind::VoidPtr;
            type_.origin_ident = None;

            if type_alias.args.is_some() {
                self.errors.push(AnalyzerError {
                    start: type_.start,
                    end: type_.end,
                    message: format!("void_ptr cannot contains arg"),
                });
            }

            return true;
        }

        // raw ptr rewrite
        if type_alias.ident == TypeKind::RawPtr(Box::new(Type::default())).to_string() {
            // extract first args to type_
            if let Some(args) = type_alias.args {
                let mut first_arg_type = args[0].clone();
                self.analyze_type(&mut first_arg_type);
                type_.kind = TypeKind::RawPtr(Box::new(first_arg_type));
            } else {
                self.errors.push(AnalyzerError {
                    start: type_.start,
                    end: type_.end,
                    message: format!("raw_ptr must contains one arg"),
                });
            }

            type_.origin_ident = None;
            type_.origin_type_kind = TypeKind::Unknown;
            return true;
        }

        // ptr rewrite
        if type_alias.ident == TypeKind::Ptr(Box::new(Type::default())).to_string() {
            if let Some(args) = type_alias.args {
                let mut first_arg_type = args[0].clone();
                self.analyze_type(&mut first_arg_type);
                type_.kind = TypeKind::Ptr(Box::new(first_arg_type));
            } else {
                self.errors.push(AnalyzerError {
                    start: type_.start,
                    end: type_.end,
                    message: format!("ptr must contains one arg"),
                });
            }

            type_.origin_ident = None;
            type_.origin_type_kind = TypeKind::Unknown;
            return true;
        }

        // all_t rewrite
        if type_alias.ident == TypeKind::AllT.to_string() {
            type_.kind = TypeKind::AllT;
            type_.origin_ident = None;
            type_.origin_type_kind = TypeKind::Unknown;
            if type_alias.args.is_some() {
                self.errors.push(AnalyzerError {
                    start: type_.start,
                    end: type_.end,
                    message: format!("all_type cannot contains arg"),
                });
            }
            return true;
        }

        // fn_t rewrite
        if type_alias.ident == TypeKind::FnT.to_string() {
            type_.kind = TypeKind::FnT;
            type_.origin_ident = None;
            type_.origin_type_kind = TypeKind::Unknown;
            if type_alias.args.is_some() {
                self.errors.push(AnalyzerError {
                    start: type_.start,
                    end: type_.end,
                    message: format!("fn_t cannot contains arg"),
                });
            }
            return true;
        }

        return false;
    }

    fn analyze_type(&mut self, type_: &mut Type) {
        match type_.kind {
            TypeKind::Alias(ref mut type_alias) => {
                let ident = type_alias.ident.clone();

                // 处理导入的全局模式别名，例如 type a = package.foo
                if let Some(ref import_as) = type_alias.import_as {
                    // 在导入表中查找对应的导入
                    for import in &self.imports {
                        if import.as_name != *import_as {
                            continue;
                        }

                        // 更新标识符指向
                        let global_pkg_ident = format!("{}.{}", import.package_ident, ident);

                        // check exists
                        if !self.symbol_table.symbol_exists_in_scope(&global_pkg_ident, GLOBAL_SCOPE_ID) {
                            self.errors.push(AnalyzerError {
                                start: type_.start,
                                end: type_.end,
                                message: format!("type alias '{}' undeclared", ident),
                            });
                            return;
                        }

                        type_alias.ident = global_pkg_ident;
                        type_alias.import_as = None;

                        break;
                    }
                } else {
                    // no import as
                    if let Some(unique_alias_ident) = self.resolve_type_alias(&type_alias.ident, self.symbol_table.current_scope_id) {
                        type_alias.ident = unique_alias_ident;
                    } else {
                        // check is special type ident
                        if self.analyze_special_type_rewrite(type_) {
                            return;
                        }

                        self.errors.push(AnalyzerError {
                            start: type_.start,
                            end: type_.end,
                            message: format!("type '{}' undeclared", ident),
                        });
                        return;
                    }
                }

                // 处理泛型参数
                if let Some(args) = &mut type_alias.args {
                    for arg in args {
                        self.analyze_type(arg);
                    }
                }
            }
            TypeKind::Union(any, ref mut elements) => {
                if !any {
                    for element in elements.iter_mut() {
                        self.analyze_type(element);
                    }
                }
            }
            TypeKind::Map(ref mut key_type, ref mut value_type) => {
                self.analyze_type(key_type);
                self.analyze_type(value_type);
            }
            TypeKind::Set(ref mut element_type) => {
                self.analyze_type(element_type);
            }
            TypeKind::Vec(ref mut element_type) => {
                self.analyze_type(element_type);
            }
            TypeKind::Chan(ref mut element_type) => {
                self.analyze_type(element_type);
            }
            TypeKind::Arr(_, ref mut element_type) => {
                self.analyze_type(element_type);
            }
            TypeKind::Tuple(ref mut elements, _align) => {
                for element in elements {
                    self.analyze_type(element);
                }
            }
            TypeKind::Ptr(ref mut value_type) => {
                self.analyze_type(value_type);
            }
            TypeKind::RawPtr(ref mut value_type) => {
                self.analyze_type(value_type);
            }
            TypeKind::Fn(ref mut fn_type) => {
                self.analyze_type(&mut fn_type.return_type);
                for param_type in &mut fn_type.param_types {
                    self.analyze_type(param_type);
                }
            }
            TypeKind::Struct(_, _, ref mut properties) => {
                for property in properties {
                    self.analyze_type(&mut property.type_);

                    // 可选的又值
                    if let Some(ref mut value) = property.value {
                        self.analyze_expr(value);

                        // value kind cannot is fndef
                        if let AstNode::FnDef(..) = value.node {
                            self.errors.push(AnalyzerError {
                                start: value.start,
                                end: value.end,
                                message: format!("struct field default value cannot be a fn def, use fn def ident instead"),
                            });
                        }
                    }
                }
            }
            _ => {
                return;
            }
        }
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

                    if let Err(e) = self
                        .symbol_table
                        .define_symbol(fndef.symbol_name.clone(), SymbolKind::Fn(fndef_mutex.clone()), fndef.symbol_start)
                    {
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

                    // global ident to symbol_table
                    if let Err(e) = self
                        .symbol_table
                        .define_symbol(var_decl.ident.clone(), SymbolKind::Var(var_decl_mutex.clone()), var_decl.symbol_start)
                    {
                        self.errors.push(AnalyzerError {
                            start: var_decl.symbol_start,
                            end: var_decl.symbol_end,
                            message: e,
                        });
                    }

                    self.analyze_type(&mut var_decl.type_);

                    // 将 vardef 转换成 assign 导入到 package init 中进行初始化
                    let assign_left = Box::new(Expr::ident(var_decl.symbol_start, var_decl.symbol_end, var_decl.ident.clone()));
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
                    if let Some(ref mut params) = type_alias.params {
                        for param in params.iter_mut() {
                            if !param.constraints.0 {
                                // 遍历所有 constraints 类型 进行 analyze
                                for constraint in &mut param.constraints.1 {
                                    self.analyze_type(constraint);
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

        return (stmt_list, self.errors.clone());
    }

    pub fn resolve_type_alias(&mut self, ident: &str, scope_id: NodeId) -> Option<String> {
        if scope_id == GLOBAL_SCOPE_ID {
            let curent_package_ident = format!("{}.{}", self.package_name, ident);

            // check global symbol map
            if self.symbol_table.symbol_exists_in_scope(&curent_package_ident, scope_id) {
                return Some(curent_package_ident);
            }

            // import x as * 产生的全局符号
            for i in &self.imports {
                if i.as_name != "*" {
                    continue;
                };

                let temp_package_ident = format!("{}.{}", i.package_ident, ident);
                if self.symbol_table.symbol_exists_in_scope(&temp_package_ident, scope_id) {
                    return Some(temp_package_ident);
                }
            }

            // builtin 全局符号，不需要进行 format 链接，直接读取 global 符号表
            if self.symbol_table.symbol_exists_in_scope(ident, scope_id) {
                return Some(ident.to_string());
            }

            return None;
        }

        let scope_option = self.symbol_table.find_scope(scope_id);
        return self.resolve_type_alias(ident, scope_option.unwrap().parent.unwrap());
    }

    pub fn analyze_global_fn(&mut self, fndef_mutex: Arc<Mutex<AstFnDef>>) {
        let mut fndef = fndef_mutex.lock().unwrap();
        fndef.is_local = false;
        if fndef.generics_params.is_some() {
            fndef.is_generics = true;
        }

        if fndef.is_tpl {
            assert!(fndef.body.len() == 0);
        }

        self.analyze_type(&mut fndef.return_type);

        // 如果 impl type 是 type alias, 则从符号表中获取当前的 type alias 的全称进行更新
        // fn vec<T>.len() -> fn vec_len(vec<T> self)
        if fndef.impl_type.kind.is_exist() {
            if matches!(fndef.impl_type.kind, TypeKind::Alias(..)) {
                let Some(impl_ident) = &fndef.impl_type.impl_ident else {
                    panic!("impl ident empty")
                };
                if let Some(unique_alias_ident) = self.resolve_type_alias(impl_ident, GLOBAL_SCOPE_ID) {
                    fndef.impl_type.impl_ident = Some(unique_alias_ident.clone());
                    // fndef.impl_type.kind
                    if let TypeKind::Alias(ref mut alias_box) = fndef.impl_type.kind {
                        alias_box.ident = unique_alias_ident.clone();
                    }
                } else {
                    self.errors.push(AnalyzerError {
                        start: fndef.symbol_start,
                        end: fndef.symbol_end,
                        message: format!("type alias '{}' undeclared", impl_ident),
                    });
                };
            }

            // 重构 params 的位置, 新增 self param
            let mut new_params = Vec::new();
            let param_type = fndef.impl_type.clone();
            let self_vardecl = VarDeclExpr {
                ident: String::from("self"),
                type_: param_type,
                be_capture: false,
                heap_ident: None,
                symbol_start: fndef.symbol_start,
                symbol_end: fndef.symbol_end,
                symbol_id: None,
            };

            new_params.push(Arc::new(Mutex::new(self_vardecl)));
            new_params.extend(fndef.params.iter().cloned());
            fndef.params = new_params;
        }

        self.symbol_table.enter_create_scope(ScopeKind::GlobalFn(fndef_mutex.clone()));

        // 函数形参处理
        for param_mutex in &fndef.params {
            let mut param = param_mutex.lock().unwrap();
            self.analyze_type(&mut param.type_);

            // 将参数添加到符号表中
            match self
                .symbol_table
                .define_symbol(param.ident.clone(), SymbolKind::Var(param_mutex.clone()), param.symbol_start)
            {
                Ok(symbol_id) => {
                    param.symbol_id = Some(symbol_id);
                }
                Err(e) => {
                    self.errors.push(AnalyzerError {
                        start: param.symbol_start,
                        end: param.symbol_end,
                        message: e,
                    });
                }
            }
        }

        if fndef.body.len() > 0 {
            self.analyze_body(&mut fndef.body);
        }

        // 将当前的 fn 添加到 global fn 的 local_children 中
        fndef.local_children = self.current_local_fn_list.clone();
        // 清空 self.current_local_fn_list
        self.current_local_fn_list.clear();

        self.symbol_table.exit_scope();
    }

    pub fn analyze_body(&mut self, body: &mut Vec<Box<Stmt>>) {
        for stmt in body {
            self.analyze_stmt(stmt);
        }
    }

    pub fn analyze_as_star_or_builtin(&mut self, ident: &str) -> Option<(NodeId, String)> {
        // import * ident
        for import in &self.imports {
            if import.as_name == "*" {
                let global_ident = format!("{}.{}", import.package_ident, ident);
                if let Some(id) = self.symbol_table.find_symbol(&global_ident, GLOBAL_SCOPE_ID) {
                    return Some((id, global_ident));
                }
            }
        }

        // builtin ident
        if let Some(id) = self.symbol_table.find_symbol(ident, GLOBAL_SCOPE_ID) {
            return Some((id, ident.to_string()));
        }

        return None;
    }

    pub fn analyze_unknown_select(&mut self, expr: &mut Box<Expr>) {
        let AstNode::SelectExpr(left, key) = &mut expr.node else { unreachable!() };

        if let AstNode::Ident(ident, symbol_id) = &mut left.node {
            // 尝试 find local or parent ident, 如果找到，将 symbol_id 添加到 Ident 中
            // symbol 可能是 parent local, 也可能是 parent fn，此时则发生闭包函数引用, 需要将 ident 改写成 env access
            if let Some(id) = self.symbol_table.lookup_symbol(ident) {
                *symbol_id = Some(id);
                return;
            }

            let current_pkg_ident = format!("{}.{}", self.package_name, ident);

            // find current package global ident
            if let Some(id) = self.symbol_table.find_symbol(&current_pkg_ident, GLOBAL_SCOPE_ID) {
                // change ident to pkg_ident
                *ident = current_pkg_ident;
                *symbol_id = Some(id);
                return;
            }

            // import package ident
            for import in &self.imports {
                if import.package_ident == *ident {
                    // 找到对应的 import
                    // 构造全局唯一标识符
                    let global_ident = format!("{}.{}", import.package_ident, key);

                    // 检查是否存在于 global import symbol 中
                    if let Some(id) = self.symbol_table.find_symbol(&global_ident, GLOBAL_SCOPE_ID) {
                        *ident = global_ident;
                        *symbol_id = Some(id);
                        return;
                    }
                }
            }

            // builtin ident or import as * 也会产生 select expr, 如果是 import *
            if let Some((id, global_ident)) = self.analyze_as_star_or_builtin(ident) {
                *ident = global_ident;
                *symbol_id = Some(id);
                return;
            }

            self.errors.push(AnalyzerError {
                start: left.start,
                end: left.end,
                message: format!("identifier '{}' undeclared", ident),
            });

            return; 
        }

        self.analyze_expr(left);
    }

    pub fn analyze_ident(&mut self, ident: &mut String, symbol_id: &mut Option<NodeId>) -> bool {
        // 尝试 find local or parent ident, 如果找到，将 symbol_id 添加到 Ident 中
        // symbol 可能是 parent local, 也可能是 parent fn，此时则发生闭包函数引用, 需要将 ident 改写成 env access
        if let Some(id) = self.symbol_table.lookup_symbol(ident) {
            *symbol_id = Some(id);
            return true;
        }

        // current package ident
        let current_pkg_ident = format!("{}.{}", self.package_name, ident);
        if let Some(id) = self.symbol_table.find_symbol(&current_pkg_ident, GLOBAL_SCOPE_ID) {
            *ident = current_pkg_ident;
            *symbol_id = Some(id);
            return true;
        }

        if let Some((id, global_ident)) = self.analyze_as_star_or_builtin(ident) {
            *ident = global_ident;
            *symbol_id = Some(id);
            return true;
        }

        return false;
    }

    pub fn analyze_match(&mut self, subject: &mut Option<Box<Expr>>, cases: &mut Vec<MatchCase>) {
        let mut subject_ident: Option<String> = None;

        if let Some(subject_expr) = subject {
            // if ident
            if let AstNode::Ident(ident, _) = &subject_expr.node {
                subject_ident = Some(ident.clone());
            }

            self.analyze_expr(subject_expr);
        }

        self.symbol_table.enter_create_scope(ScopeKind::Local);
        let cases_len = cases.len();
        for (i, case) in cases.iter_mut().enumerate() {
            let cond_list_len = case.cond_list.len();
            let mut is_cond = false;
            for cond in case.cond_list.iter_mut() {
                // default case check
                if let AstNode::Ident(ident, _symbol_id) = &cond.node {
                    if ident == "_" {
                        if cond_list_len != 1 {
                            self.errors.push(AnalyzerError {
                                start: cond.start,
                                end: cond.end,
                                message: "default case '_' conflict in a 'match' expression".to_string(),
                            });
                        }

                        if i != cases_len - 1 {
                            self.errors.push(AnalyzerError {
                                start: cond.start,
                                end: cond.end,
                                message: "default case '_' must be the last one in a 'match' expression".to_string(),
                            });
                        }

                        case.is_default = true;
                        continue;
                    }
                } else if let AstNode::Is(target_type, src) = &cond.node {
                    is_cond = true;
                }

                self.analyze_expr(cond);
            }

            if case.cond_list.len() > 1 {
                is_cond = false; // cond is logic, not is expr
            }

            if is_cond && subject_ident.is_some() {
                let Some(subject_literal) = subject_ident.clone() else { unreachable!() };
                let Some(cond_expr) = case.cond_list.first() else { unreachable!() };
                let AstNode::Is(target_type, _) = &cond_expr.node else { unreachable!() };
                case.handle_body
                    .insert(0, self.auto_as_stmt(cond_expr.start, cond_expr.end, &subject_literal, target_type));
            }

            self.symbol_table.enter_create_scope(ScopeKind::Local);
            self.analyze_body(&mut case.handle_body);
            self.symbol_table.exit_scope();
        }
        self.symbol_table.exit_scope();
    }

    pub fn analyze_async(&mut self, async_expr: &mut MacroAsyncExpr) {
        self.analyze_local_fndef(&async_expr.closure_fn);
        self.analyze_local_fndef(&async_expr.closure_fn_void);

        // closure_fn 的 fn_name 需要继承当前 fn 的 fn_name, 这样报错才会更加的精准
        let mut fndef = async_expr.closure_fn.lock().unwrap();
        let Some(global_fn_mutex) = self.symbol_table.find_global_fn() else {
            panic!("global fn not found")
        };
        let global_fn = global_fn_mutex.lock().unwrap();
        fndef.fn_name = global_fn.fn_name.clone();

        self.analyze_call(&mut async_expr.origin_call);
        if let Some(flag_expr) = &mut async_expr.flag_expr {
            self.analyze_expr(flag_expr);
        }
    }

    pub fn analyze_expr(&mut self, expr: &mut Box<Expr>) {
        match &mut expr.node {
            AstNode::Binary(_op, left, right) => {
                self.analyze_expr(left);
                self.analyze_expr(right);
            }
            AstNode::Unary(_op, expr) => {
                self.analyze_expr(expr);
            }
            AstNode::Catch(try_expr, catch_err, catch_body) => {
                self.analyze_expr(try_expr);

                self.symbol_table.enter_create_scope(ScopeKind::Local);
                self.analyze_var_decl(catch_err);
                self.analyze_body(catch_body);
                self.symbol_table.exit_scope();
            }
            AstNode::As(type_, src) => {
                self.analyze_type(type_);
                self.analyze_expr(src);
            }
            AstNode::Is(target_type, src) => {
                self.analyze_type(target_type);
                self.analyze_expr(src);
            }
            AstNode::MacroSizeof(target_type) => {
                self.analyze_type(target_type);
            }
            AstNode::MacroUla(src) => {
                self.analyze_expr(src);
            }
            AstNode::MacroReflectHash(target_type) => {
                self.analyze_type(target_type);
            }
            AstNode::MacroTypeEq(left_type, right_type) => {
                self.analyze_type(left_type);
                self.analyze_type(right_type);
            }
            AstNode::New(type_, properties) => {
                self.analyze_type(type_);
                for property in properties {
                    self.analyze_expr(&mut property.value);
                }
            }
            AstNode::StructNew(_ident, type_, properties) => {
                self.analyze_type(type_);
                for property in properties {
                    self.analyze_expr(&mut property.value);
                }
            }
            AstNode::MapNew(elements) => {
                for element in elements {
                    self.analyze_expr(&mut element.key);
                    self.analyze_expr(&mut element.value);
                }
            }
            AstNode::SetNew(elements) => {
                for element in elements {
                    self.analyze_expr(element);
                }
            }
            AstNode::TupleNew(elements) => {
                for element in elements {
                    self.analyze_expr(element);
                }
            }
            AstNode::TupleDestr(elements) => {
                for element in elements {
                    self.analyze_expr(element);
                }
            }
            AstNode::VecNew(elements, len, cap) => {
                for element in elements {
                    self.analyze_expr(element);
                }
            }
            AstNode::AccessExpr(left, key) => {
                self.analyze_expr(left);
                self.analyze_expr(key);
            }
            AstNode::SelectExpr(..) => self.analyze_unknown_select(expr),
            AstNode::Ident(ident, symbol_id) => {
                if !self.analyze_ident(ident, symbol_id) {
                    self.errors.push(AnalyzerError {
                        start: expr.start,
                        end: expr.end,
                        message: format!("identifier '{}' undeclared", ident),
                    });
                }
            }
            AstNode::Match(subject, cases) => self.analyze_match(subject, cases),
            AstNode::Call(call) => self.analyze_call(call),
            AstNode::MacroAsync(async_expr) => self.analyze_async(async_expr),
            AstNode::FnDef(fndef_mutex) => self.analyze_local_fndef(fndef_mutex),
            _ => {
                return;
            }
        }
    }

    /* if (expr->assert_type == AST_VAR_DECL) {
        analyzer_var_decl(m, expr->value, true);
    } else if (expr->assert_type == AST_EXPR_TUPLE_DESTR) {
        analyzer_var_tuple_destr(m, expr->value);
    } else {
        ANALYZER_ASSERTF(false, "var tuple destr expr type exception");
    } */
    pub fn analyze_var_tuple_destr_item(&mut self, item: &Box<Expr>) {
        match &item.node {
            AstNode::VarDecl(var_decl_mutex) => {
                self.analyze_var_decl(var_decl_mutex);
            }
            AstNode::TupleDestr(elements) => {
                self.analyze_var_tuple_destr(elements);
            }
            _ => {
                self.errors.push(AnalyzerError {
                    start: item.start,
                    end: item.end,
                    message: "var tuple destr expr type exception".to_string(),
                });
            }
        }
    }

    pub fn analyze_var_tuple_destr(&mut self, elements: &Vec<Box<Expr>>) {
        for item in elements.iter() {
            self.analyze_var_tuple_destr_item(item);
        }
    }

    pub fn analyze_call(&mut self, call: &mut AstCall) {
        self.analyze_expr(&mut call.left);

        for generics_arg in call.generics_args.iter_mut() {
            self.analyze_type(generics_arg);
        }

        for arg in call.args.iter_mut() {
            self.analyze_expr(arg);
        }
    }

    /**
     * local fn in global fn
     */
    pub fn analyze_local_fndef(&mut self, fndef_mutex: &Arc<Mutex<AstFnDef>>) {
        let mut fndef = fndef_mutex.lock().unwrap();

        // find global fn in symbol table
        let Some(global_fn_mutex) = self.symbol_table.find_global_fn() else {
            panic!("global fn not found")
        };
        fndef.global_parent = Some(global_fn_mutex.clone());
        fndef.is_local = true;

        self.current_local_fn_list.push(fndef_mutex.clone());

        // local fn 作为闭包函数, 不能进行类型扩展和泛型参数
        if fndef.impl_type.kind.is_exist() || fndef.generics_params.is_some() {
            self.errors.push(AnalyzerError {
                start: fndef.symbol_start,
                end: fndef.symbol_end,
                message: "closure fn cannot be generics or impl type alias".to_string(),
            });
        }

        // 闭包不能包含 macro ident
        if fndef.linkid.is_some() {
            self.errors.push(AnalyzerError {
                start: fndef.symbol_start,
                end: fndef.symbol_end,
                message: "closure fn cannot have #linkid label".to_string(),
            });
        }

        if fndef.is_tpl {
            self.errors.push(AnalyzerError {
                start: fndef.symbol_start,
                end: fndef.symbol_end,
                message: "closure fn cannot be template".to_string(),
            });
        }

        self.symbol_table.enter_create_scope(ScopeKind::LocalFn(fndef_mutex.clone()));

        // 形参处理
        for param_mutex in &fndef.params {
            let mut param = param_mutex.lock().unwrap();
            self.analyze_type(&mut param.type_);

            // 将参数添加到符号表中
            match self
                .symbol_table
                .define_symbol(param.ident.clone(), SymbolKind::Var(param_mutex.clone()), param.symbol_start)
            {
                Ok(symbol_id) => {
                    param.symbol_id = Some(symbol_id);
                }
                Err(e) => {
                    self.errors.push(AnalyzerError {
                        start: param.symbol_start,
                        end: param.symbol_end,
                        message: e,
                    });
                }
            }
        }

        // handle body
        self.analyze_body(&mut fndef.body);

        let mut free_var_count = 0;
        let scope = self.symbol_table.get_scope();
        for (_, free_ident) in scope.frees.iter() {
            if matches!(free_ident.kind, SymbolKind::Var(..)) {
                free_var_count += 1;
            }
        }

        self.symbol_table.exit_scope();

        // 当前函数需要编译成闭包, 所有的 call fn 改造成 call fn_var
        if free_var_count > 0 {
            fndef.is_closure = true;
        }

        // 将 fndef lambda 添加到 symbol table 中
        match self
            .symbol_table
            .define_symbol(fndef.symbol_name.clone(), SymbolKind::Fn(fndef_mutex.clone()), fndef.symbol_start)
        {
            Ok(symbol_id) => {
                fndef.symbol_id = Some(symbol_id);
            }
            Err(e) => {
                self.errors.push(AnalyzerError {
                    start: fndef.symbol_start,
                    end: fndef.symbol_end,
                    message: e,
                });
            }
        }
    }

    pub fn extract_is_expr(&mut self, cond: &Box<Expr>) -> Option<Box<Expr>> {
        if let AstNode::Is(_target_type, src) = &cond.node {
            // is src 必须是 ident 才能进行 自动 as 转换
            if let AstNode::Ident(..) = &src.node {
                return Some(cond.clone());
            }
        }

        // binary && extract
        if let AstNode::Binary(op, left, right) = &cond.node {
            if *op == ExprOp::AndAnd {
                let left_is = self.extract_is_expr(left);
                let right_is = self.extract_is_expr(right);

                // condition expr cannot contains multiple is expr
                if left_is.is_some() && right_is.is_some() {
                    self.errors.push(AnalyzerError {
                        start: cond.start,
                        end: cond.end,
                        message: "condition expr cannot contains multiple is expr".to_string(),
                    });
                }

                return if left_is.is_some() { left_is } else { right_is };
            }
        }

        return None;
    }

    pub fn auto_as_stmt(&mut self, start: usize, end: usize, subject_ident: &str, target_type: &Type) -> Box<Stmt> {
        // var x = x as T
        let var_decl = Arc::new(Mutex::new(VarDeclExpr {
            ident: subject_ident.to_string(),
            type_: target_type.clone(),
            be_capture: false,
            heap_ident: None,
            symbol_start: start,
            symbol_end: end,
            symbol_id: None,
        }));

        // 创建标识符表达式作为 as 表达式的源
        let src_expr = Box::new(Expr::ident(start, end, subject_ident.to_string()));
        let as_expr = Box::new(Expr {
            node: AstNode::As(target_type.clone(), src_expr),
            start,
            end,
            type_: Type::default(),
            target_type: Type::default(),
        });

        // 创建最终的变量定义语句
        Box::new(Stmt {
            node: AstNode::VarDef(var_decl, as_expr),
            start,
            end,
        })
    }

    pub fn analyze_if(&mut self, cond: &mut Box<Expr>, consequent: &mut Vec<Box<Stmt>>, alternate: &mut Vec<Box<Stmt>>) {
        // if has is expr push T e = e as T
        if let Some(is_expr) = self.extract_is_expr(cond) {
            assert!(matches!(is_expr.node, AstNode::Ident(..)));
            let AstNode::Ident(ident, _) = is_expr.node else { unreachable!() };
            let ast_stmt = self.auto_as_stmt(is_expr.start, is_expr.end, &ident, &is_expr.type_);
            // insert ast_stmt to consequent first
            consequent.insert(0, ast_stmt);
        }

        self.analyze_expr(cond);

        self.symbol_table.enter_create_scope(ScopeKind::Local);
        self.analyze_body(consequent);
        self.symbol_table.exit_scope();

        self.symbol_table.enter_create_scope(ScopeKind::Local);
        self.analyze_body(alternate);
        self.symbol_table.exit_scope();
    }

    pub fn analyze_stmt(&mut self, stmt: &mut Box<Stmt>) {
        match &mut stmt.node {
            AstNode::Fake(expr) => {
                self.analyze_expr(expr);
            }
            AstNode::VarDecl(var_decl_mutex) => {
                self.analyze_var_decl(var_decl_mutex);
            }
            AstNode::VarDef(var_decl_mutex, expr) => {
                self.analyze_expr(expr);
                self.analyze_var_decl(var_decl_mutex);
            }
            AstNode::VarTupleDestr(elements, expr) => {
                self.analyze_expr(expr);
                self.analyze_var_tuple_destr(elements);
            }
            AstNode::Assign(left, right) => {
                self.analyze_expr(left);
                self.analyze_expr(right);
            }
            AstNode::Call(call) => {
                self.analyze_call(call);
            }
            AstNode::Catch(try_expr, catch_err, catch_body) => {
                self.analyze_expr(try_expr);
                self.analyze_var_decl(&catch_err);
                self.symbol_table.enter_create_scope(ScopeKind::Local);
                self.analyze_body(catch_body);
                self.symbol_table.exit_scope();
            }
            AstNode::Select(cases, _has_default, _send_count, _recv_count) => {
                let len = cases.len();
                for (i, case) in cases.iter_mut().enumerate() {
                    if let Some(on_call) = &mut case.on_call {
                        self.analyze_call(on_call);
                    }
                    self.symbol_table.enter_create_scope(ScopeKind::Local);
                    if let Some(recv_var) = &case.recv_var {
                        self.analyze_var_decl(recv_var);
                    }
                    self.analyze_body(&mut case.handle_body);
                    self.symbol_table.exit_scope();

                    if case.is_default && i != len - 1 {
                        // push error
                        self.errors.push(AnalyzerError {
                            start: case.handle_body[0].start,
                            end: case.handle_body[0].end,
                            message: "default case must be the last case".to_string(),
                        });
                    }
                }
            }
            AstNode::Throw(expr) => {
                self.analyze_expr(expr);
            }
            AstNode::If(cond, consequent, alternate) => {
                self.analyze_if(cond, consequent, alternate);
            }
            AstNode::ForCond(condition, body) => {
                self.analyze_expr(condition);
                self.symbol_table.enter_create_scope(ScopeKind::Local);
                self.analyze_body(body);
                self.symbol_table.exit_scope();
            }
            AstNode::ForIterator(iterate, first, second, body) => {
                self.analyze_expr(iterate);

                self.symbol_table.enter_create_scope(ScopeKind::Local);
                self.analyze_var_decl(first);
                if let Some(second) = second {
                    self.analyze_var_decl(second);
                }
                self.analyze_body(body);
                self.symbol_table.exit_scope();
            }
            AstNode::ForTradition(init, condition, update, body) => {
                self.symbol_table.enter_create_scope(ScopeKind::Local);
                self.analyze_stmt(init);
                self.analyze_expr(condition);
                self.analyze_stmt(update);
                self.analyze_body(body);
                self.symbol_table.exit_scope();
            }
            AstNode::Return(expr) => {
                if let Some(expr) = expr {
                    self.analyze_expr(expr);
                }
            }
            AstNode::Break(expr) => {
                if let Some(expr) = expr {
                    self.analyze_expr(expr);
                }
            }
            AstNode::TypeAlias(type_alias_mutex) => {
                let mut type_alias = type_alias_mutex.lock().unwrap();
                // local type alias 不允许携带 param
                if type_alias.params.is_some() {
                    self.errors.push(AnalyzerError {
                        start: type_alias.symbol_start,
                        end: type_alias.symbol_end,
                        message: "local type alias cannot have params".to_string(),
                    });
                }

                self.analyze_type(&mut type_alias.type_);

                match self.symbol_table.define_symbol(
                    type_alias.ident.clone(),
                    SymbolKind::TypeAlias(type_alias_mutex.clone()),
                    type_alias.symbol_start,
                ) {
                    Ok(symbol_id) => {
                        type_alias.symbol_id = Some(symbol_id);
                    }
                    Err(e) => {
                        self.errors.push(AnalyzerError {
                            start: type_alias.symbol_start,
                            end: type_alias.symbol_end,
                            message: e,
                        });
                    }
                }
            }
            _ => {
                return;
            }
        }
    }

    pub fn analyze_var_decl(&mut self, var_decl_mutex: &Arc<Mutex<VarDeclExpr>>) {
        let mut var_decl = var_decl_mutex.lock().unwrap();

        self.analyze_type(&mut var_decl.type_);

        // 添加到符号表，返回值 sysmbol_id 添加到 var_decl 中, 已经包含了 redeclare check
        match self
            .symbol_table
            .define_symbol(var_decl.ident.clone(), SymbolKind::Var(var_decl_mutex.clone()), var_decl.symbol_start)
        {
            Ok(symbol_id) => {
                var_decl.symbol_id = Some(symbol_id);
            }
            Err(e) => {
                self.errors.push(AnalyzerError {
                    start: var_decl.symbol_start,
                    end: var_decl.symbol_end,
                    message: e,
                });
            }
        }
    }
}
