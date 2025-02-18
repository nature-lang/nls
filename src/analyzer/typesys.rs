use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use log::debug;

use crate::{
    analyzer::{
        common::*,
        symbol::{SymbolKind, GLOBAL_SCOPE_ID},
    },
    project::Module,
    utils::format_generics_ident,
};

use super::{
    common::{AnalyzerError, AstCall, AstNode, Expr, Stmt, Type, TypeAliasStmt, TypeFn, VarDeclExpr},
    symbol::{NodeId, SymbolTable},
};

#[derive(Debug, Clone)]
pub struct GenericSpecialFnClone {
    // default 是 none, clone 过程中, 当 global fn clone 完成后，将 clone 完成的 global fn 赋值给 global_parent
    global_parent: Option<Arc<Mutex<AstFnDef>>>,
}

impl GenericSpecialFnClone {
    pub fn deep_clone(&mut self, fn_mutex: &Arc<Mutex<AstFnDef>>) -> Arc<Mutex<AstFnDef>> {
        let fn_def = fn_mutex.lock().unwrap();
        let mut fn_def_clone = fn_def.clone();

        // type 中不包含 arc, 所以可以直接进行 clone
        fn_def_clone.type_ = fn_def.type_.clone();
        // params 中包含 arc, 所以需要解构后进行 clone
        fn_def_clone.params = fn_def
            .params
            .iter()
            .map(|param| {
                let param_clone = param.lock().unwrap().clone();
                Arc::new(Mutex::new(param_clone))
            })
            .collect();

        // 清空 generics 标识
        fn_def_clone.is_generics = false;

        // 递归进行 body 的 clone
        fn_def_clone.body = self.clone_body(&fn_def.body);
        fn_def_clone.global_parent = None;

        // 重新完善 children 和 parent 关系
        if fn_def_clone.is_local {
            assert!(self.global_parent.is_some());
            fn_def_clone.global_parent = self.global_parent.clone();
            {
                let mut global_parent = self.global_parent.as_ref().unwrap().lock().unwrap();
                let result = Arc::new(Mutex::new(fn_def_clone));
                global_parent.local_children.push(result.clone());
                return result;
            }
        } else {
            let result = Arc::new(Mutex::new(fn_def_clone));
            self.global_parent = Some(result.clone());
            return result;
        }
    }

    fn clone_body(&mut self, body: &Vec<Box<Stmt>>) -> Vec<Box<Stmt>> {
        body.iter().map(|stmt| Box::new(self.clone_stmt(stmt))).collect()
    }

    fn clone_expr(&mut self, expr: &Expr) -> Expr {
        let node = match &expr.node {
            AstNode::Literal(kind, value) => AstNode::Literal(kind.clone(), value.clone()),
            AstNode::Ident(ident, symbol_id) => AstNode::Ident(ident.clone(), symbol_id.clone()),
            AstNode::EnvAccess(index, ident, symbol_id) => AstNode::EnvAccess(*index, ident.clone(), symbol_id.clone()),

            AstNode::Binary(op, left, right) => AstNode::Binary(op.clone(), Box::new(self.clone_expr(left)), Box::new(self.clone_expr(right))),
            AstNode::Unary(op, operand) => AstNode::Unary(op.clone(), Box::new(self.clone_expr(operand))),
            AstNode::AccessExpr(left, key) => AstNode::AccessExpr(Box::new(self.clone_expr(left)), Box::new(self.clone_expr(key))),
            AstNode::VecNew(elements, len, cap) => AstNode::VecNew(
                elements.iter().map(|e| Box::new(self.clone_expr(e))).collect(),
                len.as_ref().map(|e| Box::new(self.clone_expr(e))),
                cap.as_ref().map(|e| Box::new(self.clone_expr(e))),
            ),
            AstNode::ArrayNew(elements) => AstNode::ArrayNew(elements.iter().map(|e| Box::new(self.clone_expr(e))).collect()),
            AstNode::VecAccess(type_, left, index) => AstNode::VecAccess(type_.clone(), Box::new(self.clone_expr(left)), Box::new(self.clone_expr(index))),
            AstNode::EmptyCurlyNew => AstNode::EmptyCurlyNew,

            AstNode::MapNew(elements) => AstNode::MapNew(
                elements
                    .iter()
                    .map(|e| MapElement {
                        key: Box::new(self.clone_expr(&e.key)),
                        value: Box::new(self.clone_expr(&e.value)),
                    })
                    .collect(),
            ),
            AstNode::MapAccess(key_type, value_type, left, key) => AstNode::MapAccess(
                key_type.clone(),
                value_type.clone(),
                Box::new(self.clone_expr(left)),
                Box::new(self.clone_expr(key)),
            ),
            AstNode::StructNew(ident, type_, props) => AstNode::StructNew(
                ident.clone(),
                type_.clone(),
                props
                    .iter()
                    .map(|p| StructNewProperty {
                        type_: p.type_.clone(),
                        key: p.key.clone(),
                        value: Box::new(self.clone_expr(&p.value)),
                        start: p.start,
                        end: p.end,
                    })
                    .collect(),
            ),
            AstNode::StructSelect(instance, key, property) => AstNode::StructSelect(
                Box::new(self.clone_expr(instance)),
                key.clone(),
                TypeStructProperty {
                    type_: property.type_.clone(),
                    key: property.key.clone(),
                    start: property.start,
                    end: property.end,
                    value: Some(Box::new(self.clone_expr(&property.value.as_ref().unwrap()))),
                },
            ),
            AstNode::TupleNew(elements) => AstNode::TupleNew(elements.iter().map(|e| Box::new(self.clone_expr(e))).collect()),
            AstNode::TupleDestr(elements) => AstNode::TupleDestr(elements.iter().map(|e| Box::new(self.clone_expr(e))).collect()),
            AstNode::TupleAccess(type_, left, index) => AstNode::TupleAccess(type_.clone(), Box::new(self.clone_expr(left)), *index),

            AstNode::SetNew(elements) => AstNode::SetNew(elements.iter().map(|e| Box::new(self.clone_expr(e))).collect()),
            AstNode::Call(call) => AstNode::Call(self.clone_call(call)),
            AstNode::MacroAsync(async_expr) => AstNode::MacroAsync(MacroAsyncExpr {
                closure_fn: self.deep_clone(&async_expr.closure_fn),
                closure_fn_void: self.deep_clone(&async_expr.closure_fn_void),
                origin_call: Box::new(self.clone_call(&async_expr.origin_call)),
                flag_expr: async_expr.flag_expr.as_ref().map(|e| Box::new(self.clone_expr(e))),
                return_type: async_expr.return_type.clone(),
            }),

            AstNode::FnDef(fn_def_mutex) => AstNode::FnDef(self.deep_clone(fn_def_mutex)),
            AstNode::New(type_, props) => AstNode::New(
                type_.clone(),
                props
                    .iter()
                    .map(|p| StructNewProperty {
                        type_: p.type_.clone(),
                        key: p.key.clone(),
                        value: Box::new(self.clone_expr(&p.value)),
                        start: p.start,
                        end: p.end,
                    })
                    .collect(),
            ),
            AstNode::As(type_, src) => AstNode::As(type_.clone(), Box::new(self.clone_expr(src))),
            AstNode::Is(type_, src) => AstNode::Is(type_.clone(), Box::new(self.clone_expr(src))),
            AstNode::MatchIs(type_) => AstNode::MatchIs(type_.clone()),
            AstNode::Catch(try_expr, catch_err, catch_body) => AstNode::Catch(
                Box::new(self.clone_expr(try_expr)),
                Arc::new(Mutex::new(catch_err.lock().unwrap().clone())),
                self.clone_body(catch_body),
            ),
            AstNode::Match(subject, cases) => AstNode::Match(subject.as_ref().map(|s| Box::new(self.clone_expr(s))), self.clone_match_cases(cases)),

            AstNode::MacroSizeof(type_) => AstNode::MacroSizeof(type_.clone()),
            AstNode::MacroUla(src) => AstNode::MacroUla(Box::new(self.clone_expr(src))),
            AstNode::MacroReflectHash(type_) => AstNode::MacroReflectHash(type_.clone()),
            AstNode::MacroTypeEq(left, right) => AstNode::MacroTypeEq(left.clone(), right.clone()),
            AstNode::MacroDefault => AstNode::MacroDefault,

            AstNode::ArrayAccess(type_, left, index) => AstNode::ArrayAccess(type_.clone(), Box::new(self.clone_expr(left)), Box::new(self.clone_expr(index))),

            AstNode::SelectExpr(left, key) => AstNode::SelectExpr(Box::new(self.clone_expr(left)), key.clone()),
            _ => expr.node.clone(),
        };

        Expr {
            start: expr.start,
            end: expr.end,
            type_: expr.type_.clone(),
            target_type: expr.target_type.clone(),
            node,
            err: expr.err,
        }
    }

    fn clone_stmt(&mut self, stmt: &Stmt) -> Stmt {
        let node = match &stmt.node {
            AstNode::Fake(expr) => AstNode::Fake(Box::new(self.clone_expr(expr))),
            AstNode::VarDecl(var_decl) => AstNode::VarDecl(Arc::new(Mutex::new(var_decl.lock().unwrap().clone()))),
            AstNode::VarDef(var_decl, right) => AstNode::VarDef(Arc::new(Mutex::new(var_decl.lock().unwrap().clone())), Box::new(self.clone_expr(right))),
            AstNode::VarTupleDestr(elements, expr) => {
                let new_elements: Vec<Box<Expr>> = elements.iter().map(|e| Box::new(self.clone_expr(e))).collect();
                AstNode::VarTupleDestr(new_elements, Box::new(self.clone_expr(expr)))
            }
            AstNode::Assign(left, right) => AstNode::Assign(Box::new(self.clone_expr(left)), Box::new(self.clone_expr(right))),
            AstNode::If(condition, consequent, alternate) => {
                AstNode::If(Box::new(self.clone_expr(condition)), self.clone_body(consequent), self.clone_body(alternate))
            }
            AstNode::ForCond(condition, body) => AstNode::ForCond(Box::new(self.clone_expr(condition)), self.clone_body(body)),
            AstNode::ForIterator(iterate, first, second, body) => AstNode::ForIterator(
                Box::new(self.clone_expr(iterate)),
                Arc::new(Mutex::new(first.lock().unwrap().clone())),
                second.as_ref().map(|s| Arc::new(Mutex::new(s.lock().unwrap().clone()))),
                self.clone_body(body),
            ),
            AstNode::ForTradition(init, cond, update, body) => AstNode::ForTradition(
                Box::new(self.clone_stmt(init)),
                Box::new(self.clone_expr(cond)),
                Box::new(self.clone_stmt(update)),
                self.clone_body(body),
            ),
            AstNode::FnDef(fn_def_mutex) => AstNode::FnDef(self.deep_clone(fn_def_mutex)),
            AstNode::Throw(expr) => AstNode::Throw(Box::new(self.clone_expr(expr))),
            AstNode::Return(expr_opt) => AstNode::Return(expr_opt.as_ref().map(|e| Box::new(self.clone_expr(e)))),
            AstNode::Call(call) => AstNode::Call(self.clone_call(call)),
            AstNode::Continue => AstNode::Continue,
            AstNode::Break(expr_opt) => AstNode::Break(expr_opt.as_ref().map(|e| Box::new(self.clone_expr(e)))),
            AstNode::Catch(try_expr, catch_err, catch_body) => AstNode::Catch(
                Box::new(self.clone_expr(try_expr)),
                Arc::new(Mutex::new(catch_err.lock().unwrap().clone())),
                self.clone_body(catch_body),
            ),
            AstNode::Select(cases, has_default, send_count, recv_count) => {
                AstNode::Select(self.clone_select_cases(cases), *has_default, *send_count, *recv_count)
            }
            AstNode::Match(subject, cases) => AstNode::Match(subject.as_ref().map(|s| Box::new(self.clone_expr(s))), self.clone_match_cases(cases)),

            AstNode::TryCatch(try_expr, catch_err, catch_body) => AstNode::TryCatch(
                Box::new(self.clone_expr(try_expr)),
                Arc::new(Mutex::new(catch_err.lock().unwrap().clone())),
                self.clone_body(catch_body),
            ),

            AstNode::TypeAlias(alias) => AstNode::TypeAlias(Arc::new(Mutex::new(alias.lock().unwrap().clone()))),
            _ => stmt.node.clone(),
        };

        Stmt {
            start: stmt.start,
            end: stmt.end,
            node,
        }
    }

    fn clone_match_cases(&mut self, cases: &Vec<MatchCase>) -> Vec<MatchCase> {
        cases
            .iter()
            .map(|case| MatchCase {
                cond_list: case.cond_list.iter().map(|e| Box::new(self.clone_expr(e))).collect(),
                is_default: case.is_default,
                handle_body: self.clone_body(&case.handle_body),
                start: case.start,
                end: case.end,
            })
            .collect()
    }

    fn clone_select_cases(&mut self, cases: &Vec<SelectCase>) -> Vec<SelectCase> {
        cases
            .iter()
            .map(|case| SelectCase {
                on_call: case.on_call.as_ref().map(|call| self.clone_call(call)),
                recv_var: case.recv_var.as_ref().map(|var| Arc::new(Mutex::new(var.lock().unwrap().clone()))),
                is_recv: case.is_recv,
                is_default: case.is_default,
                handle_body: self.clone_body(&case.handle_body),
                start: case.start,
                end: case.end,
            })
            .collect()
    }

    fn clone_call(&mut self, call: &AstCall) -> AstCall {
        AstCall {
            return_type: call.return_type.clone(),
            left: Box::new(self.clone_expr(&call.left)),
            generics_args: call.generics_args.clone(),
            args: call.args.iter().map(|e| Box::new(self.clone_expr(e))).collect(),
            spread: call.spread,
        }
    }
}

#[derive(Debug)]
pub struct Typesys<'a> {
    symbol_table: &'a mut SymbolTable,
    module: &'a mut Module,
    current_fn_mutex: Arc<Mutex<AstFnDef>>,
    worklist: Vec<Arc<Mutex<AstFnDef>>>,
    generics_args_stack: Vec<HashMap<String, Type>>,
    be_caught: bool,
    break_target_types: Vec<Type>,
    errors: Vec<AnalyzerError>,
}

impl<'a> Typesys<'a> {
    pub fn new(symbol_table: &'a mut SymbolTable, module: &'a mut Module) -> Self {
        assert!(symbol_table.current_scope_id == GLOBAL_SCOPE_ID);

        Self {
            symbol_table,
            module,
            worklist: Vec::new(),
            generics_args_stack: Vec::new(),
            current_fn_mutex: Arc::new(Mutex::new(AstFnDef::default())),
            be_caught: false,
            break_target_types: Vec::new(),
            errors: Vec::new(),
        }
    }

    fn finalize_type(&mut self, t: Type, origin_ident: Option<String>, origin_type_kind: TypeKind) -> Type {
        let mut result = t.clone();
        result.origin_ident = origin_ident;
        result.origin_type_kind = origin_type_kind;
        result.status = ReductionStatus::Done;
        result.kind = self.cross_kind_trans(&result.kind);

        return result;
    }

    fn reduction_type_alias(&mut self, t: Type) -> Result<Type, AnalyzerError> {
        let TypeKind::Alias(alias) = &t.kind else { unreachable!() };
        let start = t.start;
        let end = t.end;

        // 获取类型别名的标识符和参数
        let impl_ident = alias.ident.clone();

        if alias.symbol_id.is_none() {
            return Err(AnalyzerError {
                start: 0,
                end: 0,
                message: format!("type alias '{}' symbol_id not found", alias.ident),
            });
        }

        let symbol = self.symbol_table.get_symbol(alias.symbol_id.unwrap()).unwrap(); // 确保找到的符号是类型别名

        let type_alias_stmt_mutex = match symbol.kind.clone() {
            SymbolKind::TypeAlias(stmt) => stmt,
            _ => {
                return Err(AnalyzerError {
                    start,
                    end,
                    message: format!("'{}' is not a type", symbol.ident),
                });
            }
        };

        let mut type_alias = match type_alias_stmt_mutex.try_lock() {
            Ok(guard) => guard,
            Err(_) => return Ok(t), // 无法获取锁,说明存在递归调用,返回原始类型
        };

        let mut impl_args = Vec::new();

        // 处理带参数的类型别名
        if type_alias.params.len() > 0 {
            if alias.args.is_none() {
                return Err(AnalyzerError {
                    start,
                    end,
                    message: format!("type alias '{}' need param", alias.ident),
                });
            }

            let args = alias.args.as_ref().unwrap();

            if args.len() != type_alias.params.len() {
                return Err(AnalyzerError {
                    start,
                    end,
                    message: format!("type alias '{}' param not match", alias.ident),
                });
            }

            // 对每个参数进行类型归约和约束检查
            for (i, undo_arg_type) in args.iter().enumerate() {
                let arg_type = self.reduction_type(undo_arg_type.clone())?;
                impl_args.push(arg_type.clone());

                let param = &mut type_alias.params[i];

                // 处理约束条件
                if !param.constraints.0 {
                    for constraint in param.constraints.1.iter_mut() {
                        *constraint = self.reduction_type(constraint.clone())?;
                    }
                }

                if !self.union_type_contains(&param.constraints, &arg_type) {
                    return Err(AnalyzerError {
                        start,
                        end,
                        message: format!("type alias '{}' param constraint not match", alias.ident),
                    });
                }
            }

            // 处理类型参数栈
            let mut args_table: HashMap<String, Type> = HashMap::new();
            for (i, undo_arg_type) in args.iter().enumerate() {
                let arg_type = self.reduction_type(undo_arg_type.clone())?;
                let param = &type_alias.params[i];

                args_table.insert(param.ident.clone(), arg_type.clone());
            }

            self.generics_args_stack.push(args_table);

            // 对右值类型进行归约
            let mut alias_value_type = type_alias.type_expr.clone();
            alias_value_type = self.reduction_type(alias_value_type)?;

            if self.generics_args_stack.len() > 0 {
                self.generics_args_stack.pop();
            }

            // 设置实现标识符和参数
            alias_value_type.impl_ident = Some(impl_ident);
            alias_value_type.impl_args = impl_args;
            return Ok(alias_value_type);
        }

        // 检查是否已完成归约
        if type_alias.type_expr.status == ReductionStatus::Done {
            type_alias.type_expr.impl_ident = Some(impl_ident);
            type_alias.type_expr.impl_args = impl_args;
            return Ok(type_alias.type_expr.clone());
        }

        // 处理正在进行归约的情况(避免循环依赖)
        if type_alias.type_expr.status == ReductionStatus::Doing {
            return Ok(t);
        }

        // 进行类型归约
        type_alias.type_expr.status = ReductionStatus::Doing;
        type_alias.type_expr = self.reduction_type(type_alias.type_expr.clone())?;
        type_alias.type_expr.impl_ident = Some(impl_ident);
        type_alias.type_expr.impl_args = impl_args;

        return Ok(type_alias.type_expr.clone());
    }

    fn reduction_complex_type(&mut self, t: Type) -> Result<Type, AnalyzerError> {
        let mut result = t.clone();

        let kind_str = result.kind.to_string();

        match &mut result.kind {
            // 处理指针类型
            TypeKind::Ptr(value_type) | TypeKind::RawPtr(value_type) => {
                *value_type = Box::new(self.reduction_type(*value_type.clone())?);
                result.impl_ident = value_type.impl_ident.clone();
                result.impl_args = value_type.impl_args.clone();
            }

            // 处理数组类型
            TypeKind::Arr(_, element_type) => {
                *element_type = Box::new(self.reduction_type(*element_type.clone())?);
            }

            // 处理通道类型
            TypeKind::Chan(element_type) => {
                *element_type = Box::new(self.reduction_type(*element_type.clone())?);
                result.impl_ident = Some(kind_str);
                result.impl_args = vec![*element_type.clone()];
            }

            // 处理向量类型
            TypeKind::Vec(element_type) => {
                *element_type = Box::new(self.reduction_type(*element_type.clone())?);
                result.impl_ident = Some(kind_str);
                result.impl_args = vec![*element_type.clone()];
            }

            // 处理映射类型
            TypeKind::Map(key_type, value_type) => {
                *key_type = Box::new(self.reduction_type(*key_type.clone())?);
                *value_type = Box::new(self.reduction_type(*value_type.clone())?);

                // 检查键类型是否合法
                if !Type::is_map_key_type(&key_type.kind) {
                    return Err(AnalyzerError {
                        start: t.start,
                        end: t.end,
                        message: format!("type '{}' not support as map key", key_type),
                    });
                }

                result.impl_ident = Some(kind_str);
                result.impl_args = vec![*key_type.clone(), *value_type.clone()];
            }

            // 处理集合类型
            TypeKind::Set(element_type) => {
                *element_type = Box::new(self.reduction_type(*element_type.clone())?);

                // 检查元素类型是否合法
                if !Type::is_map_key_type(&element_type.kind) {
                    return Err(AnalyzerError {
                        start: t.start,
                        end: t.end,
                        message: format!("type '{}' not support as set element", element_type),
                    });
                }

                result.impl_ident = Some("set".to_string());
                result.impl_args = vec![*element_type.clone()];
            }

            // 处理元组类型
            TypeKind::Tuple(elements, align) => {
                assert!(!elements.is_empty(), "tuple element empty");

                let mut max_align = 0;
                for element_type in elements.iter_mut() {
                    *element_type = self.reduction_type(element_type.clone())?;
                    let element_align = Type::alignof(&element_type.kind);
                    max_align = max_align.max(element_align);
                }

                *align = max_align;
            }
            TypeKind::Fn(type_fn) => {
                type_fn.return_type = self.reduction_type(type_fn.return_type.clone())?;

                for param_type in type_fn.param_types.iter_mut() {
                    *param_type = self.reduction_type(param_type.clone())?;
                }
            }

            // 处理结构体类型
            TypeKind::Struct(_ident, align, properties) => {
                let mut max_align = 0;

                for property in properties.iter_mut() {
                    if !property.type_.kind.is_unknown() {
                        property.type_ = self.reduction_type(property.type_.clone())?;
                    }

                    if let Some(right_value) = &mut property.value {
                        match self.infer_right_expr(right_value, property.type_.clone()) {
                            Ok(right_type) => {
                                if property.type_.kind.is_unknown() {
                                    // 如果属性类型未知,则 必须能够推导其类型
                                    if !self.type_confirm(&right_type) {
                                        self.errors_push(
                                            right_value.start,
                                            right_value.end,
                                            format!("struct property '{}' type not confirmed", property.key),
                                        );
                                    }

                                    property.type_ = right_type;
                                }
                            }
                            Err(e) => {
                                self.errors_push(e.start, e.end, e.message);
                            }
                        }
                    }

                    let item_align = Type::alignof(&property.type_.kind);
                    max_align = max_align.max(item_align);

                    if !self.type_confirm(&property.type_) {
                        self.errors_push(
                            property.type_.start,
                            property.type_.end,
                            format!("struct property '{}' type not confirmed", property.key),
                        );
                    }
                }

                *align = max_align;
            }

            _ => panic!("unknown type={}", result),
        }

        Ok(result)
    }

    pub fn reduction_type(&mut self, mut t: Type) -> Result<Type, AnalyzerError> {
        let origin_ident = t.origin_ident.clone();
        let origin_type_kind = t.origin_type_kind.clone();

        if t.err {
            return Err(AnalyzerError {
                start: 0,
                end: 0,
                message: format!("type {} already has error", t),
            });
        }

        if t.kind.is_unknown() {
            return Ok(t);
        }

        if t.status == ReductionStatus::Done {
            return Ok(self.finalize_type(t, origin_ident, origin_type_kind));
        }

        match &mut t.kind {
            TypeKind::Alias(_) => {
                let result = self.reduction_type_alias(t)?;
                return Ok(self.finalize_type(result, origin_ident, origin_type_kind));
            }
            TypeKind::Param(ident) => {
                if self.generics_args_stack.is_empty() {
                    return Ok(self.finalize_type(t, origin_ident, origin_type_kind));
                }

                let args_table = self.generics_args_stack.last().unwrap();
                let result = self.reduction_type(args_table.get(ident).unwrap().clone())?;
                return Ok(self.finalize_type(result, origin_ident, origin_type_kind));
            }
            TypeKind::Union(any, elements) => {
                if *any {
                    return Ok(self.finalize_type(t, origin_ident, origin_type_kind));
                }

                for element in elements {
                    *element = self.reduction_type(element.clone())?;
                }

                return Ok(self.finalize_type(t, origin_ident, origin_type_kind));
            }
            _ => {
                if Type::is_origin_type(&t.kind) {
                    let mut result = t.clone();
                    result.impl_ident = Some(t.kind.to_string());
                    return Ok(self.finalize_type(result, origin_ident, origin_type_kind));
                }

                if Type::is_reduction_type(&t.kind) {
                    let result = self.reduction_complex_type(t)?;
                    return Ok(self.finalize_type(result, origin_ident, origin_type_kind));
                }

                panic!("cannot parser type: {}", t);
            }
        }
    }

    pub fn infer_as_expr(&mut self, expr: &mut Box<Expr>) -> Result<Type, AnalyzerError> {
        // 先还原目标类型
        let AstNode::As(target_type, src) = &mut expr.node else { unreachable!() };
        *target_type = self.reduction_type(target_type.clone())?;

        // 推导源表达式类型, 如果存在错误则停止后续比较, 直接返回错误
        let src_type = self.infer_expr(src, target_type.clone())?;

        // 处理raw指针转换的特殊情况
        if matches!(src_type.kind, TypeKind::RawPtr(..)) {
            if !matches!(target_type.kind, TypeKind::VoidPtr) {
                return Err(AnalyzerError {
                    start: expr.start,
                    end: expr.end,
                    message: format!("{} can only as void_ptr", src_type),
                });
            }
            return Ok(target_type.clone());
        }

        // 如果源类型和目标类型相同，直接返回目标类型
        if self.type_compare(&src_type, target_type) {
            return Ok(target_type.clone());
        }

        // 处理联合类型转换
        if let TypeKind::Union(any, elements) = &src_type.kind {
            if matches!(target_type.kind, TypeKind::Union(..)) {
                return Err(AnalyzerError {
                    start: expr.start,
                    end: expr.end,
                    message: "union to union type is not supported".to_string(),
                });
            }

            // 检查目标类型是否包含在联合类型中
            if !self.union_type_contains(&(*any, elements.clone()), &target_type) {
                return Err(AnalyzerError {
                    start: expr.start,
                    end: expr.end,
                    message: format!("type = {} not contains in union type", target_type),
                });
            }
            return Ok(target_type.clone());
        }

        // string -> list u8
        if matches!(src_type.kind, TypeKind::String) && Type::is_list_u8(&target_type.kind) {
            return Ok(target_type.clone());
        }

        // list u8 -> string
        if Type::is_list_u8(&src_type.kind) && matches!(target_type.kind, TypeKind::String) {
            return Ok(target_type.clone());
        }

        // void_ptr相关转换
        if !Type::is_float(&src_type.kind) && matches!(target_type.kind, TypeKind::VoidPtr) {
            return Ok(target_type.clone());
        }

        if matches!(src_type.kind, TypeKind::VoidPtr) && !Type::is_float(&target_type.kind) {
            return Ok(target_type.clone());
        }

        // 检查目标类型是否可以进行类型转换
        if !Type::can_type_casting(&target_type.kind) {
            return Err(AnalyzerError {
                start: expr.start,
                end: expr.end,
                message: format!("cannot casting to '{}'", target_type),
            });
        }

        Ok(target_type.clone())
    }

    pub fn infer_match(
        &mut self,
        subject: &mut Option<Box<Expr>>,
        cases: &mut Vec<MatchCase>,
        target_type: Type,
        start: usize,
        end: usize,
    ) -> Result<Type, AnalyzerError> {
        // 默认 subject_type 为 bool 类型(用于无 subject 的情况)
        let mut subject_type = Type::new(TypeKind::Bool);

        // 如果存在 subject,推导其类型, 如果推倒错误则停止后续推倒，因为无法识别具体的类型
        if let Some(subject_expr) = subject {
            subject_type = self.infer_right_expr(subject_expr, Type::default())?;

            // 确保 subject 类型已确定
            if !self.type_confirm(&subject_type) {
                return Err(AnalyzerError {
                    start: subject_expr.start,
                    end: subject_expr.end,
                    message: "match subject type not confirm".to_string(),
                });
            }
        }

        // 将目标类型加入 break_target_types 栈
        self.break_target_types.push(target_type.clone());

        // 用于跟踪联合类型的匹配情况
        let mut union_types = HashMap::new();
        let mut has_default = false;

        // 遍历所有 case
        for case in cases {
            if case.is_default {
                has_default = true;
            } else {
                // 处理每个条件表达式
                for cond_expr in case.cond_list.iter_mut() {
                    // 对于联合类型,只能使用 is 匹配
                    if matches!(subject_type.kind, TypeKind::Union(..)) {
                        if !matches!(cond_expr.node, AstNode::MatchIs(..)) {
                            return Err(AnalyzerError {
                                start: cond_expr.start,
                                end: cond_expr.end,
                                message: "match 'union type' only support 'is' assert".to_string(),
                            });
                        }
                    }

                    // 处理 is 类型匹配
                    if let AstNode::MatchIs(_target_type) = cond_expr.node.clone() {
                        if !matches!(subject_type.kind, TypeKind::Union(..)) {
                            return Err(AnalyzerError {
                                start: cond_expr.start,
                                end: cond_expr.end,
                                message: "only union type can use is assert".to_string(),
                            });
                        }

                        let cond_type = self.infer_right_expr(cond_expr, Type::default())?;
                        assert!(matches!(cond_type.kind, TypeKind::Bool));

                        let AstNode::MatchIs(target_type) = &cond_expr.node else { unreachable!() };

                        // 记录已匹配的类型, 最终可以判断 match 是否匹配了所有分支
                        union_types.insert(target_type.hash(), true);
                    } else {
                        // 普通值匹配,推导条件表达式类型
                        self.infer_right_expr(cond_expr, subject_type.clone())?;
                    }
                }
            }

            // 推导 case 处理体
            self.infer_body(&mut case.handle_body);
        }

        // 检查 default case
        if !has_default {
            // 对于非 any 的联合类型,检查是否所有可能的类型都已匹配
            if let TypeKind::Union(any, elements) = &subject_type.kind {
                if !any {
                    for element_type in elements {
                        if !union_types.contains_key(&element_type.hash()) {
                            return Err(AnalyzerError {
                                start,
                                end,
                                message: format!(
                                    "match expression lacks a default case '_' and union element type lacks, for example 'is {}'",
                                    element_type
                                ),
                            });
                        }
                    }
                } else {
                    return Err(AnalyzerError {
                        start,
                        end,
                        message: "match expression lacks a default case '_'".to_string(),
                    });
                }
            } else {
                return Err(AnalyzerError {
                    start,
                    end,
                    message: "match expression lacks a default case '_'".to_string(),
                });
            }
        }

        // 弹出目标类型
        return Ok(self.break_target_types.pop().unwrap());
    }

    pub fn infer_struct_properties(
        &mut self,
        type_properties: &mut Vec<TypeStructProperty>,
        properties: &mut Vec<StructNewProperty>,
    ) -> Result<Vec<StructNewProperty>, AnalyzerError> {
        // 用于跟踪已经处理过的属性
        let mut exists = HashMap::new();

        // 处理显式指定的属性
        for property in properties.iter_mut() {
            // 在类型定义中查找对应的属性
            let expect_property = type_properties.iter().find(|p| p.key == property.key).ok_or_else(|| AnalyzerError {
                start: property.start,
                end: property.end,
                message: format!("not found property '{}'", property.key),
            })?;

            exists.insert(property.key.clone(), true);

            // 推导属性值的类型
            if let Err(e) = self.infer_right_expr(&mut property.value, expect_property.type_.clone()) {
                self.errors_push(e.start, e.end, e.message);
            }

            // 冗余属性类型(用于计算size)
            property.type_ = expect_property.type_.clone();
        }

        // 处理默认值
        let mut result = properties.clone();

        // 遍历类型定义中的所有属性
        for type_prop in type_properties.iter() {
            // 如果属性已经被显式指定或没有默认值,则跳过
            if exists.contains_key(&type_prop.key) || type_prop.value.is_none() {
                continue;
            }

            exists.insert(type_prop.key.clone(), true);

            // 添加默认值属性
            result.push(StructNewProperty {
                type_: type_prop.type_.clone(),
                key: type_prop.key.clone(),
                value: type_prop.value.clone().unwrap(),
                start: type_prop.start,
                end: type_prop.end,
            });
        }

        // 检查所有必需的属性是否都已赋值
        for type_prop in type_properties.iter() {
            if exists.contains_key(&type_prop.key) {
                continue;
            }

            // 检查是否是必须赋值的类型
            if Type::must_assign_value(&type_prop.type_.kind) {
                return Err(AnalyzerError {
                    start: type_prop.start,
                    end: type_prop.end,
                    message: format!("property '{}' type '{}' must assign value", type_prop.key, type_prop.type_),
                });
            }
        }

        Ok(result)
    }

    pub fn infer_binary(&mut self, op: ExprOp, left: &mut Box<Expr>, right: &mut Box<Expr>, _infer_target_type: Type) -> Result<Type, AnalyzerError> {
        // 先推导左操作数的类型
        let left_type = self.infer_right_expr(left, Type::default())?;

        // 推导右操作数的类型
        let right_type = if !matches!(left_type.kind, TypeKind::Union(..)) {
            // 如果左操作数不是联合类型,则基于左操作数类型进行隐式类型转换
            self.infer_right_expr(right, left_type.clone())?
        } else {
            // 否则独立推导右操作数类型
            self.infer_right_expr(right, Type::default())?
        };

        // 检查左右操作数类型是否一致
        if !self.type_compare(&left_type, &right_type) {
            return Err(AnalyzerError {
                start: left.start,
                end: right.end,
                message: format!("binary type inconsistency: left={}, right={}", left_type, right_type),
            });
        }

        // 处理数值类型运算
        if Type::is_number(&left_type.kind) {
            // 检查右操作数也必须是数值类型
            if !Type::is_number(&right_type.kind) {
                return Err(AnalyzerError {
                    start: right.start,
                    end: right.end,
                    message: format!(
                        "binary operator '{}' only support number operand, actual '{} {} {}'",
                        op, left_type, op, right_type
                    ),
                });
            }
        }

        // 处理字符串类型运算
        if matches!(left_type.kind, TypeKind::String) {
            // 检查右操作数也必须是字符串类型
            if !matches!(right_type.kind, TypeKind::String) {
                return Err(AnalyzerError {
                    start: right.start,
                    end: right.end,
                    message: format!(
                        "binary operator '{}' only support string operand, actual '{} {} {}'",
                        op, left_type, op, right_type
                    ),
                });
            }
        }

        // 处理位运算操作符
        if op.is_integer() {
            // 检查操作数必须是整数类型
            if !Type::is_integer(&left_type.kind) || !Type::is_integer(&right_type.kind) {
                return Err(AnalyzerError {
                    start: left.start,
                    end: right.end,
                    message: format!("binary operator '{}' only integer operand", op),
                });
            }
        }

        // 返回结果类型
        if op.is_arithmetic() {
            // 算术运算返回左操作数类型
            Ok(left_type)
        } else if op.is_logic() {
            // 逻辑运算返回布尔类型
            Ok(Type::new(TypeKind::Bool))
        } else {
            // 未知运算符
            Err(AnalyzerError {
                start: left.start,
                end: right.end,
                message: format!("unknown operator '{}'", op),
            })
        }
    }

    pub fn infer_unary(&mut self, op: ExprOp, operand: &mut Box<Expr>) -> Result<Type, AnalyzerError> {
        // 处理逻辑非运算符
        if op == ExprOp::Not {
            // 对任何类型都可以进行布尔转换
            return Ok(self.infer_right_expr(operand, Type::new(TypeKind::Bool))?);
        }

        // 获取操作数的类型
        let operand_type = self.infer_right_expr(operand, Type::default())?;

        // 处理负号运算符
        if op == ExprOp::Neg && !Type::is_number(&operand_type.kind) {
            return Err(AnalyzerError {
                start: operand.start,
                end: operand.end,
                message: "neg operand must applies to int or float type".to_string(),
            });
        }

        // 处理取地址运算符 &
        if op == ExprOp::La {
            // 检查是否是字面量或函数调用
            if matches!(operand.node, AstNode::Literal(..) | AstNode::Call(..)) {
                return Err(AnalyzerError {
                    start: operand.start,
                    end: operand.end,
                    message: "cannot load address of an literal or call".to_string(),
                });
            }

            // 检查是否是联合类型
            if matches!(operand_type.kind, TypeKind::Union(..)) {
                return Err(AnalyzerError {
                    start: operand.start,
                    end: operand.end,
                    message: "cannot load address of an union type".to_string(),
                });
            }

            return Ok(Type::raw_ptr_of(operand_type));
        }

        // 处理不安全取地址运算符 @unsafe_la
        if op == ExprOp::UnsafeLa {
            // 检查是否是字面量或函数调用
            if matches!(operand.node, AstNode::Literal(..) | AstNode::Call(..)) {
                return Err(AnalyzerError {
                    start: operand.start,
                    end: operand.end,
                    message: "cannot safe load address of an literal or call".to_string(),
                });
            }

            // 检查是否是联合类型
            if matches!(operand_type.kind, TypeKind::Union(..)) {
                return Err(AnalyzerError {
                    start: operand.start,
                    end: operand.end,
                    message: "cannot safe load address of an union type".to_string(),
                });
            }

            return Ok(Type::ptr_of(operand_type));
        }

        // 处理解引用运算符 *
        if op == ExprOp::Ia {
            // 检查是否是指针类型
            match operand_type.kind {
                TypeKind::Ptr(value_type) | TypeKind::RawPtr(value_type) => {
                    return Ok(*value_type);
                }
                _ => {
                    return Err(AnalyzerError {
                        start: operand.start,
                        end: operand.end,
                        message: format!("cannot dereference non-pointer type '{}'", operand_type),
                    });
                }
            }
        }

        // 其他情况直接返回操作数类型
        Ok(operand_type)
    }

    pub fn infer_ident(&mut self, ident: &mut String, symbol_id_option: &mut Option<NodeId>, start: usize, end: usize) -> Result<Type, AnalyzerError> {
        let Some(symbol_id) = symbol_id_option else {
            return Err(AnalyzerError {
                start: 0,
                end: 0,
                message: format!("ident '{}' symbol_id is None", ident),
            });
        };

        let symbol = self.symbol_table.get_symbol(*symbol_id).unwrap();
        assert!(symbol.ident == *ident);
        let mut symbol_kind = symbol.kind.clone();
        let is_local = symbol.defined_in != GLOBAL_SCOPE_ID;

        // 判断符号是否是 local symbol
        if is_local {
            if let Some(hash) = self.current_fn_mutex.lock().unwrap().generics_args_hash {
                let new_ident = format_generics_ident(ident.clone(), hash);
                if let Some(new_symbol_id) = symbol.generics_id_map.get(&new_ident) {
                    *ident = new_ident;
                    *symbol_id = *new_symbol_id;

                    symbol_kind = self.symbol_table.get_symbol(*symbol_id).unwrap().kind.clone();
                } else {
                    panic!("generics symbol rewrite failed, new ident '{}' not found", new_ident);
                }
            }
        }

        match symbol_kind {
            SymbolKind::Var(var_decl) => {
                let mut var_decl = var_decl.lock().unwrap();
                var_decl.type_ = self.reduction_type(var_decl.type_.clone())?;

                if var_decl.type_.kind.is_unknown() {
                    return Err(AnalyzerError {
                        start,
                        end,
                        message: "unknown type".to_string(),
                    });
                }
                assert!(var_decl.type_.kind.is_exist());

                return Ok(var_decl.type_.clone());
            }
            SymbolKind::Fn(fndef) => {
                return self.infer_fn_decl(fndef.clone());
            }
            _ => {
                return Err(AnalyzerError {
                    start,
                    end,
                    message: format!("unknown symbol kind: {:?}", symbol_kind),
                });
            }
        }
    }

    pub fn infer_vec_new(&mut self, expr: &mut Box<Expr>, infer_target_type: Type) -> Result<Type, AnalyzerError> {
        let AstNode::VecNew(elements, _, _) = &mut expr.node else { unreachable!() };

        // 如果目标类型是数组，则将 VecNew 重写为 ArrayNew
        if let TypeKind::Arr(length, element_type) = &infer_target_type.kind {
            // 重写表达式节点为 ArrayNew
            expr.node = AstNode::ArrayNew(elements.clone());
            let AstNode::ArrayNew(elements) = &mut expr.node else { unreachable!() };

            let result = Type::undo_new(TypeKind::Arr(length.clone(), element_type.clone()));

            // 如果数组为空，直接返回目标类型
            if elements.is_empty() {
                return self.reduction_type(result);
            }

            // 对所有元素进行类型推导
            for element in elements {
                if let Err(e) = self.infer_right_expr(element, *element_type.clone()) {
                    self.errors_push(e.start, e.end, e.message);
                }
            }

            return self.reduction_type(infer_target_type);
        }

        // 处理向量类型
        let mut element_type = Type::default(); // TYPE_UNKNOWN

        // 如果目标类型是向量，使用其元素类型
        if let TypeKind::Vec(target_element_type) = &infer_target_type.kind {
            element_type = *target_element_type.clone();
        }

        // 如果向量为空，必须能从目标类型确定元素类型
        if elements.is_empty() {
            if !self.type_confirm(&element_type) {
                return Err(AnalyzerError {
                    start: expr.start,
                    end: expr.end,
                    message: "vec element type not confirm".to_string(),
                });
            }
            return self.reduction_type(Type::undo_new(TypeKind::Vec(Box::new(element_type))));
        }

        // 如果元素类型未知，使用第一个元素的类型进行推导
        if element_type.kind.is_unknown() {
            element_type = self.infer_right_expr(&mut elements[0], Type::default())?;
        }

        // 对所有元素进行类型推导和检查
        for element in elements.iter_mut() {
            if let Err(e) = self.infer_right_expr(element, element_type.clone()) {
                self.errors_push(e.start, e.end, e.message);
            }
        }

        // 返回最终的向量类型
        self.reduction_type(Type::undo_new(TypeKind::Vec(Box::new(element_type))))
    }

    pub fn infer_map_new(&mut self, elements: &mut Vec<MapElement>, infer_target_type: Type, start: usize, end: usize) -> Result<Type, AnalyzerError> {
        // 创建一个新的 Map 类型，初始化 key 和 value 类型为未知
        let mut key_type = Type::default(); // TYPE_UNKNOWN
        let mut value_type = Type::default(); // TYPE_UNKNOWN

        // 如果有目标类型且是Map类型，使用目标类型的key和value类型
        if let TypeKind::Map(target_key_type, target_value_type) = &infer_target_type.kind {
            key_type = *target_key_type.clone();
            value_type = *target_value_type.clone();
        }

        // 如果 Map 为空，必须能从目标类型确定 key 和 value 类型
        if elements.is_empty() {
            if !self.type_confirm(&key_type) {
                return Err(AnalyzerError {
                    start,
                    end,
                    message: "map key type not confirm".to_string(),
                });
            }
            if !self.type_confirm(&value_type) {
                return Err(AnalyzerError {
                    start,
                    end,
                    message: "map value type not confirm".to_string(),
                });
            }
            return self.reduction_type(Type::undo_new(TypeKind::Map(Box::new(key_type), Box::new(value_type))));
        }

        // 如果类型未知，使用第一个元素的类型进行推导
        if key_type.kind.is_unknown() {
            key_type = self.infer_right_expr(&mut elements[0].key, Type::default())?;
            value_type = self.infer_right_expr(&mut elements[0].value, Type::default())?;
        }

        // 对所有元素进行类型推导和检查
        for element in elements.iter_mut() {
            if let Err(e) = self.infer_right_expr(&mut element.key, key_type.clone()) {
                self.errors_push(e.start, e.end, e.message);
            }

            if let Err(e) = self.infer_right_expr(&mut element.value, value_type.clone()) {
                self.errors_push(e.start, e.end, e.message);
            }
        }

        // 返回最终的Map类型
        self.reduction_type(Type::undo_new(TypeKind::Map(Box::new(key_type), Box::new(value_type))))
    }

    pub fn infer_set_new(&mut self, elements: &mut Vec<Box<Expr>>, infer_target_type: Type, start: usize, end: usize) -> Result<Type, AnalyzerError> {
        // 创建一个新的 Set 类型
        let mut element_type = Type::default(); // TYPE_UNKNOWN

        // 如果有目标类型且是 Set 类型，使用目标类型的元素类型
        if let TypeKind::Set(target_element_type) = &infer_target_type.kind {
            element_type = *target_element_type.clone();
        }

        // 如果集合为空，必须能从目标类型确定元素类型
        if elements.is_empty() {
            if !self.type_confirm(&element_type) {
                return Err(AnalyzerError {
                    start,
                    end,
                    message: "empty set element type not confirm".to_string(),
                });
            }
            return self.reduction_type(Type::undo_new(TypeKind::Set(Box::new(element_type))));
        }

        // 如果元素类型未知，使用第一个元素的类型进行推导
        if element_type.kind.is_unknown() {
            element_type = self.infer_right_expr(&mut elements[0], Type::default())?;
        }

        // 对所有元素进行类型推导和检查
        for element in elements.iter_mut() {
            if let Err(e) = self.infer_right_expr(element, element_type.clone()) {
                self.errors_push(e.start, e.end, e.message);
            }
        }

        // 返回最终的Set类型
        self.reduction_type(Type::undo_new(TypeKind::Set(Box::new(element_type))))
    }

    pub fn infer_tuple_new(&mut self, elements: &mut Vec<Box<Expr>>, infer_target_type: Type, start: usize, end: usize) -> Result<Type, AnalyzerError> {
        // 检查元组元素不能为空
        if elements.is_empty() {
            return Err(AnalyzerError {
                start,
                end,
                message: "tuple elements empty".to_string(),
            });
        }

        // 收集所有元素的类型
        let mut element_types = Vec::new();
        for (i, expr) in elements.iter_mut().enumerate() {
            let element_target_type = if let TypeKind::Tuple(target_element_types, _) = &infer_target_type.kind {
                target_element_types[i].clone()
            } else {
                Type::default()
            };

            // 推导每个元素的类型
            let expr_type = self.infer_right_expr(expr, element_target_type)?;

            // 检查元素类型是否已确定
            if !self.type_confirm(&expr_type) {
                return Err(AnalyzerError {
                    start,
                    end,
                    message: "tuple element type cannot be confirmed".to_string(),
                });
            }

            element_types.push(expr_type);
        }

        // 创建并返回元组类型
        self.reduction_type(Type::undo_new(TypeKind::Tuple(element_types, 0))) // 0 是对齐字节数，这里暂时使用0
    }

    pub fn infer_access_expr(&mut self, expr: &mut Box<Expr>) -> Result<Type, AnalyzerError> {
        let AstNode::AccessExpr(left, key) = &mut expr.node else { unreachable!() };

        // 推导左侧表达式的类型
        let left_type = self.infer_right_expr(left, Type::default())?;

        // 处理 Map 类型访问
        if let TypeKind::Map(key_type, value_type) = &left_type.kind {
            // 推导 key 表达式的类型
            self.infer_right_expr(key, *key_type.clone())?;

            // 重写为 MapAccess 节点
            expr.node = AstNode::MapAccess(*key_type.clone(), *value_type.clone(), left.clone(), key.clone());

            return Ok(*value_type.clone());
        }

        // 处理 Vec 和 String 类型访问
        if matches!(left_type.kind, TypeKind::Vec(_) | TypeKind::String) {
            // key 必须是整数类型
            self.infer_right_expr(key, Type::new(TypeKind::Int))?;

            let element_type = if let TypeKind::Vec(element_type) = &left_type.kind {
                *element_type.clone()
            } else {
                Type::new(TypeKind::Uint8) // String 的元素类型是 uint8
            };

            // 重写为 VecAccess 节点
            expr.node = AstNode::VecAccess(left_type.clone(), left.clone(), key.clone());

            return Ok(element_type);
        }

        // 处理 Array 类型访问
        if let TypeKind::Arr(_, element_type) = &left_type.kind {
            // key 必须是整数类型
            self.infer_right_expr(key, Type::new(TypeKind::Int))?;

            // 重写为 ArrayAccess 节点
            expr.node = AstNode::ArrayAccess(*element_type.clone(), left.clone(), key.clone());

            return Ok(*element_type.clone());
        }

        // 处理 Tuple 类型访问
        if let TypeKind::Tuple(elements, _) = &left_type.kind {
            // key 必须是整数字面量
            self.infer_right_expr(key, Type::new(TypeKind::Int))?;

            // 获取索引值
            let index: u64 = if let AstNode::Literal(kind, value) = &key.node {
                if !Type::is_integer(kind) {
                    return Err(AnalyzerError {
                        start: key.start,
                        end: key.end,
                        message: "tuple index must be integer literal".to_string(),
                    });
                }
                value.parse::<u64>().unwrap_or(u64::MAX)
            } else {
                return Err(AnalyzerError {
                    start: key.start,
                    end: key.end,
                    message: "tuple index must be immediate value".to_string(),
                });
            };

            // 检查索引是否越界
            if index >= elements.len() as u64 {
                return Err(AnalyzerError {
                    start: key.start,
                    end: key.end,
                    message: format!("tuple index {} out of range", index),
                });
            }

            let element_type = elements[index as usize].clone();

            // 重写为 TupleAccess 节点
            expr.node = AstNode::TupleAccess(element_type.clone(), left.clone(), index);

            return Ok(element_type);
        }

        // 不支持的类型访问
        Err(AnalyzerError {
            start: expr.start,
            end: expr.end,
            message: format!("access only support map/vec/string/array/tuple, cannot '{}'", left_type),
        })
    }

    pub fn infer_select_expr(&mut self, expr: &mut Box<Expr>) -> Result<Type, AnalyzerError> {
        let AstNode::SelectExpr(left, key) = &mut expr.node else { unreachable!() };

        // 先推导左侧表达式的类型
        let left_type = self.infer_right_expr(left, Type::default())?;

        // 处理自动解引用 - 如果是指针类型且指向结构体，则获取结构体类型
        let mut deref_type = match &left_type.kind {
            TypeKind::Ptr(value_type) | TypeKind::RawPtr(value_type) => {
                if matches!(value_type.kind, TypeKind::Struct(..)) {
                    *value_type.clone()
                } else {
                    left_type.clone()
                }
            }
            _ => left_type.clone(),
        };

        // 处理结构体类型的属性访问
        if let TypeKind::Struct(_, _, type_properties) = &mut deref_type.kind {
            // 查找属性
            if let Some(property) = type_properties.iter_mut().find(|p| p.key == *key) {
                let property_type = self.reduction_type(property.type_.clone())?;
                property.type_ = property_type.clone();

                // select -> struct select
                // StructSelect(Box<Expr>, String, StructNewProperty), // (instance, key, property)
                expr.node = AstNode::StructSelect(
                    left.clone(),
                    key.clone(),
                    TypeStructProperty {
                        type_: property_type.clone(),
                        key: property.key.clone(),
                        value: property.value.clone(),
                        start: property.start,
                        end: property.end,
                    },
                );

                return Ok(property_type);
            } else {
                // 如果找不到属性，报错
                return Err(AnalyzerError {
                    start: expr.start,
                    end: expr.end,
                    message: format!("type struct '{}' no property '{}'", deref_type.origin_ident.unwrap_or("_".to_string()), key),
                });
            }
        }

        // 如果不是结构体类型，报错
        Err(AnalyzerError {
            start: expr.start,
            end: expr.end,
            message: format!("type '{}' no property {}", left_type, key),
        })
    }

    pub fn infer_async(&mut self, expr: &mut Box<Expr>) -> Result<Type, AnalyzerError> {
        let AstNode::MacroAsync(async_expr) = &mut expr.node else { unreachable!() };
        let start = expr.start;
        let end = expr.end;

        // 处理 flag_expr，如果为 None 则创建一个默认值 0
        if async_expr.flag_expr.is_none() {
            async_expr.flag_expr = Some(Box::new(Expr {
                start,
                end,
                type_: Type::default(),
                target_type: Type::default(),
                node: AstNode::Literal(TypeKind::Int, "0".to_string()),
                err: false,
            }));
        }

        // 推导 flag_expr 类型
        if let Some(flag_expr) = &mut async_expr.flag_expr {
            if let Err(e) = self.infer_right_expr(flag_expr, Type::new(TypeKind::Int)) {
                self.errors_push(e.start, e.end, e.message);
            }
        }

        // 检查原始调用
        let fn_type = self.infer_right_expr(&mut async_expr.origin_call.left, Type::default())?;

        if let TypeKind::Fn(type_fn) = fn_type.kind {
            async_expr.return_type = type_fn.return_type.clone();

            self.infer_call_args(&mut async_expr.origin_call, *type_fn);
        } else {
            return Err(AnalyzerError {
                start: expr.start,
                end: expr.end,
                message: "async expression must call a fn".to_string(),
            });
        }

        // 构造异步调用
        let first_arg = if async_expr.return_type.kind == TypeKind::Void && async_expr.origin_call.args.is_empty() {
            // 如果没有返回值和参数，直接使用原始函数
            let mut left = async_expr.origin_call.left.clone();

            if let TypeKind::Fn(type_fn) = &mut left.type_.kind {
                type_fn.errable = true;
            }

            // 清空闭包函数体以避免推导异常
            {
                async_expr.closure_fn.lock().unwrap().body.clear();
                async_expr.closure_fn_void.lock().unwrap().body.clear();
            }

            left
        } else {
            // 需要使用闭包包装
            let _ = self.infer_fn_decl(async_expr.closure_fn.clone());
            let _ = self.infer_fn_decl(async_expr.closure_fn_void.clone());

            let closure_fn = if async_expr.return_type.kind == TypeKind::Void {
                // 使用 void 版本的闭包
                async_expr.closure_fn.lock().unwrap().body.clear();
                async_expr.closure_fn_void.clone()
            } else {
                async_expr.closure_fn.clone()
            };

            let type_ = closure_fn.lock().unwrap().type_.clone();

            Box::new(Expr {
                start,
                end,
                type_: type_.clone(),
                target_type: type_,
                node: AstNode::FnDef(closure_fn.clone()),
                err: false,
            })
        };

        // typesys 阶段需要保证所有的 ident 都包含 symbol_id, 需要从 global scope 中找到 async 对应的 symbol_id
        let symbol_id = self.symbol_table.find_symbol_id(&"async".to_string(), GLOBAL_SCOPE_ID);

        // 构造 async 调用
        let call = AstCall {
            return_type: Type::default(),
            left: Box::new(Expr::ident(start, end, "async".to_string(), symbol_id)),
            args: vec![first_arg, async_expr.flag_expr.clone().unwrap()],
            generics_args: vec![async_expr.return_type.clone()],
            spread: false,
        };

        expr.node = AstNode::Call(call);

        return Ok(self.infer_right_expr(expr, Type::default())?);
    }

    pub fn infer_expr(&mut self, expr: &mut Box<Expr>, infer_target_type: Type) -> Result<Type, AnalyzerError> {
        if expr.type_.kind.is_exist() {
            return Ok(expr.type_.clone());
        }

        return match &mut expr.node {
            AstNode::As(_, _) => self.infer_as_expr(expr),
            AstNode::Catch(try_expr, catch_err_mutex, catch_body) => {
                self.be_caught = true;
                let try_expr_type = self.infer_right_expr(try_expr, Type::default())?;
                self.be_caught = false;

                {
                    let mut errort = Type::errort(self.symbol_table);
                    errort = self.reduction_type(errort)?;
                    let mut catch_err = catch_err_mutex.lock().unwrap();
                    catch_err.type_ = errort;
                }

                self.rewrite_var_decl(catch_err_mutex.clone());

                self.break_target_types.push(try_expr_type);
                self.infer_body(catch_body);
                return Ok(self.break_target_types.pop().unwrap());
            }
            AstNode::Match(subject, cases) => self.infer_match(subject, cases, infer_target_type, expr.start, expr.end),
            AstNode::MatchIs(target_type) => {
                *target_type = self.reduction_type(target_type.clone())?;
                return Ok(Type::new(TypeKind::Bool));
            }
            AstNode::Is(target_type, src) => {
                let src_type = self.infer_right_expr(src, Type::default())?;

                *target_type = self.reduction_type(target_type.clone())?;
                if !matches!(src_type.kind, TypeKind::Union(..)) {
                    return Err(AnalyzerError {
                        start: expr.start,
                        end: expr.end,
                        message: format!("{} cannot use 'is' operator", src_type),
                    });
                }

                return Ok(Type::new(TypeKind::Bool));
            }
            AstNode::MacroSizeof(target_type) => {
                *target_type = self.reduction_type(target_type.clone())?;
                return Ok(Type::new(TypeKind::Int));
            }
            AstNode::MacroReflectHash(target_type) => {
                *target_type = self.reduction_type(target_type.clone())?;
                Ok(Type::new(TypeKind::Int))
            }
            AstNode::MacroUla(src) => {
                let src_type = self.infer_right_expr(src, Type::default())?;
                return Ok(Type::ptr_of(src_type));
            }
            AstNode::MacroDefault => {
                return Ok(infer_target_type);
            }
            AstNode::New(type_, properties) => {
                *type_ = self.reduction_type(type_.clone())?;
                if let TypeKind::Struct(_, _, type_properties) = &mut type_.kind {
                    *properties = self.infer_struct_properties(type_properties, properties)?;
                } else {
                    return Err(AnalyzerError {
                        start: expr.start,
                        end: expr.end,
                        message: "cannot use 'new' operator on non-struct type".to_string(),
                    });
                }

                return self.reduction_type(Type::ptr_of(type_.clone()));
            }
            AstNode::Binary(op, left, right) => self.infer_binary(op.clone(), left, right, infer_target_type),
            AstNode::Unary(op, operand) => self.infer_unary(op.clone(), operand),
            AstNode::Ident(ident, symbol_id) => self.infer_ident(ident, symbol_id, expr.start, expr.end),
            AstNode::VecNew(..) => self.infer_vec_new(expr, infer_target_type),
            AstNode::EmptyCurlyNew => {
                // 必须通过 target_type 引导才能推断出具体的类型, 所以 target_type kind 必须存在
                match &infer_target_type.kind {
                    TypeKind::Map(_, _) => {
                        expr.node = AstNode::MapNew(Vec::new());
                    }
                    TypeKind::Set(_) => {
                        // change expr kind to set new
                        expr.node = AstNode::SetNew(Vec::new());
                    }
                    _ => {
                        return Err(AnalyzerError {
                            start: expr.start,
                            end: expr.end,
                            message: format!("empty curly new cannot ref type {}", infer_target_type),
                        });
                    }
                }

                return Ok(infer_target_type);
            }
            AstNode::MapNew(elements) => self.infer_map_new(elements, infer_target_type, expr.start, expr.end),
            AstNode::SetNew(elements) => self.infer_set_new(elements, infer_target_type, expr.start, expr.end),
            AstNode::TupleNew(elements) => self.infer_tuple_new(elements, infer_target_type, expr.start, expr.end),
            AstNode::StructNew(_ident, type_, properties) => {
                *type_ = self.reduction_type(type_.clone())?;

                if let TypeKind::Struct(_, _, type_properties) = &mut type_.kind {
                    *properties = self.infer_struct_properties(type_properties, properties)?;
                } else {
                    return Err(AnalyzerError {
                        start: expr.start,
                        end: expr.end,
                        message: format!("cannot use 'new' operator on non-struct type {}", type_),
                    });
                }

                return Ok(type_.clone());
            }
            AstNode::AccessExpr(..) => self.infer_access_expr(expr),
            AstNode::SelectExpr(..) => self.infer_select_expr(expr),
            AstNode::Call(call) => self.infer_call(call, infer_target_type, expr.start, expr.end),
            AstNode::MacroAsync(_) => self.infer_async(expr),
            AstNode::FnDef(fndef) => self.infer_fn_decl(fndef.clone()),
            AstNode::Literal(kind, _value) => self.reduction_type(Type::new(kind.clone())),
            AstNode::EnvAccess(_, unique_ident, symbol_id_option) => {
                return self.infer_ident(unique_ident, symbol_id_option, expr.start, expr.end);
            }
            _ => panic!("unknown operand {:?}", expr.node),
        };
    }

    fn cross_kind_trans(&mut self, kind: &TypeKind) -> TypeKind {
        match kind {
            TypeKind::Float => TypeKind::Float64,
            TypeKind::Int => TypeKind::Int64,
            TypeKind::Uint => TypeKind::Uint64,
            _ => kind.clone(),
        }
    }

    fn literal_integer_casting(&mut self, expr: &mut Box<Expr>, target_type: Type) -> Result<(), AnalyzerError> {
        if let AstNode::Literal(kind, value) = &mut expr.node {
            let mut target_kind = target_type.kind.clone();

            if matches!(target_kind, TypeKind::VoidPtr) {
                target_kind = TypeKind::Uint;
            }

            if !Type::is_integer(&kind) {
                return Err(AnalyzerError {
                    start: expr.start,
                    end: expr.end,
                    message: format!("integer literal type inconsistency: expect={}, actual={}", target_type, expr.type_),
                });
            }

            let i = if value.starts_with("0x") {
                i64::from_str_radix(&value[2..], 16)
            } else if value.starts_with("0b") {
                i64::from_str_radix(&value[2..], 2)
            } else if value.starts_with("0o") {
                i64::from_str_radix(&value[2..], 8)
            } else {
                value.parse::<i64>()
            }
            .map_err(|_| AnalyzerError {
                start: expr.start,
                end: expr.end,
                message: "invalid integer literal".to_string(),
            })?;

            let target_kind = self.cross_kind_trans(&target_kind);
            if !self.integer_range_check(&target_kind, i) {
                return Err(AnalyzerError {
                    start: expr.start,
                    end: expr.end,
                    message: "integer out of range".to_string(),
                });
            }

            *kind = target_kind;
            expr.type_.kind = target_type.kind;

            Ok(())
        } else {
            Err(AnalyzerError {
                start: expr.start,
                end: expr.end,
                message: "integer casting only support literal".to_string(),
            })
        }
    }

    fn literal_float_casting(&mut self, expr: &mut Box<Expr>, target_type: Type) -> Result<(), AnalyzerError> {
        // 检查是否为字面量表达式
        if let AstNode::Literal(kind, value) = &mut expr.node {
            let target_kind = target_type.kind.clone();

            // 检查源类型是否为数字类型
            if !Type::is_number(kind) {
                return Err(AnalyzerError {
                    start: expr.start,
                    end: expr.end,
                    message: format!("type inconsistency: '{}' cannot casting float", expr.type_),
                });
            }

            // 将字符串解析为浮点数
            let f = value.parse::<f64>().map_err(|_| AnalyzerError {
                start: expr.start,
                end: expr.end,
                message: "invalid float literal".to_string(),
            })?;

            // 获取目标类型的标准形式(处理类型别名等)
            let target_kind = self.cross_kind_trans(&target_kind);

            // 检查值是否在目标类型的范围内
            if !self.float_range_check(&target_kind, f) {
                return Err(AnalyzerError {
                    start: expr.start,
                    end: expr.end,
                    message: "float out of range".to_string(),
                });
            }

            // 更新类型信息
            *kind = target_kind;
            expr.type_.kind = target_type.kind;

            Ok(())
        } else {
            // 如果不是字面量表达式，返回错误
            Err(AnalyzerError {
                start: expr.start,
                end: expr.end,
                message: "float casting only support literal".to_string(),
            })
        }
    }

    fn integer_range_check(&self, kind: &TypeKind, value: i64) -> bool {
        match kind {
            TypeKind::Int8 => value >= i8::MIN as i64 && value <= i8::MAX as i64,
            TypeKind::Int16 => value >= i16::MIN as i64 && value <= i16::MAX as i64,
            TypeKind::Int32 => value >= i32::MIN as i64 && value <= i32::MAX as i64,
            TypeKind::Int64 => value >= i64::MIN as i64 && value <= i64::MAX as i64,
            TypeKind::Uint8 => value >= 0 && value <= u8::MAX as i64,
            TypeKind::Uint16 => value >= 0 && value <= u16::MAX as i64,
            TypeKind::Uint32 => value >= 0 && value <= u32::MAX as i64,
            TypeKind::Uint64 => value >= 0,
            _ => false,
        }
    }

    fn float_range_check(&self, kind: &TypeKind, value: f64) -> bool {
        match kind {
            TypeKind::Float32 => value >= f32::MIN as f64 && value <= f32::MAX as f64,
            TypeKind::Float64 => true, // f64范围内都可以
            _ => false,
        }
    }

    fn can_assign_to_union(&self, type_: &Type) -> bool {
        match &type_.kind {
            TypeKind::Union(..) => false,
            TypeKind::Void => false,
            _ => true,
        }
    }

    fn union_type_contains(&mut self, union: &(bool, Vec<Type>), sub: &Type) -> bool {
        if union.0 {
            return true;
        }

        union.1.iter().any(|item| self.type_compare(item, sub))
    }

    pub fn infer_right_expr(&mut self, expr: &mut Box<Expr>, mut target_type: Type) -> Result<Type, AnalyzerError> {
        if expr.type_.kind.is_unknown() {
            let t = self.infer_expr(expr, target_type.clone())?;
            target_type = self.reduction_type(target_type.clone())?;

            expr.type_ = self.reduction_type(t)?;
            expr.target_type = target_type.clone();
        }

        if target_type.kind.is_unknown() {
            return Ok(expr.type_.clone());
        }

        if (Type::is_integer(&target_type.kind) || matches!(target_type.kind, TypeKind::VoidPtr)) && matches!(expr.node, AstNode::Literal(..)) {
            self.literal_integer_casting(expr, target_type.clone())?;
        }

        // 处理浮点数字面量
        if Type::is_float(&target_type.kind) && matches!(expr.node, AstNode::Literal(..)) {
            self.literal_float_casting(expr, target_type.clone())?;
        }

        if matches!(target_type.kind, TypeKind::Union(..)) && self.can_assign_to_union(&expr.type_) {
            let TypeKind::Union(any, elements) = &target_type.kind else { unreachable!() };
            if !self.union_type_contains(&(*any, elements.clone()), &expr.type_) {
                return Err(AnalyzerError {
                    start: expr.start,
                    end: expr.end,
                    message: format!("union type not contains '{}'", expr.type_),
                });
            }

            // expr 改成成 union 类型
            expr.node = AstNode::As(target_type.clone(), expr.clone());
            expr.type_ = target_type.clone();
            expr.target_type = target_type.clone();
        }

        // 最后进行类型比较
        if !self.type_compare(&target_type, &expr.type_) {
            return Err(AnalyzerError {
                start: expr.start,
                end: expr.end,
                message: format!("type inconsistency: expect={}, actual={}", target_type, expr.type_),
            });
        }

        Ok(expr.type_.clone())
    }

    pub fn infer_left_expr(&mut self, expr: &mut Box<Expr>) -> Result<Type, AnalyzerError> {
        let type_result = match &mut expr.node {
            // 标识符
            AstNode::Ident(ident, symbol_id) => self.infer_ident(ident, symbol_id, expr.start, expr.end),

            // 元组解构
            AstNode::TupleDestr(elements) => {
                let mut element_types = Vec::new();
                for element in elements.iter_mut() {
                    let element_type = self.infer_left_expr(element)?;
                    assert!(element_type.kind.is_exist());

                    element_types.push(element_type);
                }
                self.reduction_type(Type::undo_new(TypeKind::Tuple(element_types, 0)))
            }

            // 访问表达式
            AstNode::AccessExpr(..) => self.infer_access_expr(expr),

            // 选择表达式
            AstNode::SelectExpr(..) => self.infer_select_expr(expr),

            // 环境变量访问
            AstNode::EnvAccess(_, ident, symbol_id_option) => self.infer_ident(ident, symbol_id_option, expr.start, expr.end),

            // 函数调用
            AstNode::Call(call) => self.infer_call(call, Type::default(), expr.start, expr.end),

            _ => {
                // 对于不能作为左值的表达式，添加错误
                Err(AnalyzerError {
                    start: expr.start,
                    end: expr.end,
                    message: "operand cannot be used as left value".to_string(),
                })
            }
        };

        return match type_result {
            Ok(t) => {
                if !t.kind.is_exist() {
                    assert!(false, "infer left type not exist");
                }
                expr.type_ = t.clone();
                Ok(t.clone())
            }
            Err(e) => {
                return Err(e);
            }
        };
    }

    pub fn infer_vardef(&mut self, var_decl_mutex: &Arc<Mutex<VarDeclExpr>>, right_expr: &mut Box<Expr>) -> Result<(), AnalyzerError> {
        {
            let mut var_decl = var_decl_mutex.lock().unwrap();
            var_decl.type_ = self.reduction_type(var_decl.type_.clone())?;
        }

        // 重写变量声明(处理泛型情况下的变量名重写)
        self.rewrite_var_decl(var_decl_mutex.clone());

        // 检查变量类型是否为void(非参数类型的情况下)
        {
            let var_decl = var_decl_mutex.lock().unwrap();
            if !matches!(var_decl.type_.origin_type_kind, TypeKind::Param(..)) && matches!(var_decl.type_.kind, TypeKind::Void) {
                return Err(AnalyzerError {
                    start: var_decl.symbol_start,
                    end: var_decl.symbol_end,
                    message: "cannot assign to void".to_string(),
                });
            }
        }

        // 获取变量声明的类型用于右值表达式的类型推导
        let mut var_decl = var_decl_mutex.lock().unwrap();

        // 推导右值表达式的类型
        let right_type = self.infer_right_expr(right_expr, var_decl.type_.clone())?;

        // 检查右值类型是否为void
        if matches!(right_type.kind, TypeKind::Void) {
            return Err(AnalyzerError {
                start: right_expr.start,
                end: right_expr.end,
                message: "cannot assign void to var".to_string(),
            });
        }

        if matches!(var_decl.type_.kind, TypeKind::Unknown) {
            // 检查右值类型是否已确定
            if !self.type_confirm(&right_type) {
                return Err(AnalyzerError {
                    start: right_expr.start,
                    end: right_expr.end,
                    message: "type inference error, right type not confirmed".to_string(),
                });
            }

            // 使用右值类型作为变量类型
            var_decl.type_ = right_type;
        }

        Ok(())
    }

    pub fn infer_var_tuple_destr(&mut self, elements: &mut Vec<Box<Expr>>, right_type: Type, start: usize, end: usize) -> Result<(), AnalyzerError> {
        // check length
        // right_type.kind
        let TypeKind::Tuple(type_elements, _type_align) = right_type.kind else {
            unreachable!()
        };

        if type_elements.len() != elements.len() {
            return Err(AnalyzerError {
                start,
                end,
                message: format!("tuple length mismatch, expect {}, got {}", type_elements.len(), elements.len()),
            });
        }

        // 遍历按顺序对比类型,并且顺便 rewrite
        for (i, expr) in elements.iter_mut().enumerate() {
            let target_type = type_elements[i].clone();
            if !self.type_confirm(&target_type) {
                self.errors_push(expr.start, expr.end, format!("tuple operand i = {} type not confirm", i));
                continue;
            }

            assert!(target_type.kind.is_exist());
            expr.type_ = target_type.clone();

            // 递归处理
            if let AstNode::VarDecl(var_decl_mutex) = &expr.node {
                {
                    let mut var_decl = var_decl_mutex.lock().unwrap();
                    assert!(var_decl.symbol_id.is_some());

                    var_decl.type_ = target_type.clone();
                    assert!(var_decl.type_.kind.is_exist());
                }

                self.rewrite_var_decl(var_decl_mutex.clone());
            } else if let AstNode::TupleDestr(sub_elements) = &mut expr.node {
                if let Err(e) = self.infer_var_tuple_destr(sub_elements, target_type, expr.start, expr.end) {
                    self.errors_push(e.start, e.end, e.message);
                }
            } else {
                self.errors_push(expr.start, expr.end, format!("var tuple destr mut var or tuple_destr"));
            }
        }

        Ok(())
    }

    pub fn infer_catch(
        &mut self,
        try_expr: &mut Box<Expr>,
        catch_err_mutex: &Arc<Mutex<VarDeclExpr>>,
        catch_body: &mut Vec<Box<Stmt>>,
    ) -> Result<(), AnalyzerError> {
        self.be_caught = true;

        let right_type = self.infer_right_expr(try_expr, Type::default())?;

        self.be_caught = false;

        let mut errort = Type::errort(self.symbol_table);

        // reduction errort
        {
            errort = self.reduction_type(errort)?;
            let mut catch_err = catch_err_mutex.lock().unwrap();
            catch_err.type_ = errort;
        }

        self.rewrite_var_decl(catch_err_mutex.clone());

        // set break target types
        self.break_target_types.push(right_type);
        self.infer_body(catch_body);
        self.break_target_types.pop().unwrap();

        Ok(())
    }

    pub fn generics_impl_args_hash(&mut self, args: &Vec<Type>) -> Result<u64, AnalyzerError> {
        let mut hash_str = "fn".to_string();

        // 遍历所有泛型参数类型
        for arg in args {
            // 对每个参数类型进行还原
            let reduced_type = self.reduction_type(arg.clone())?;
            // 将类型的哈希值添加到字符串中
            hash_str.push_str(&reduced_type.hash().to_string());
        }

        // 计算最终的哈希值
        let mut hasher = DefaultHasher::new();
        hash_str.hash(&mut hasher);
        Ok(hasher.finish())
    }

    pub fn infer_select_call_rewrite(&mut self, call: &mut AstCall) -> Result<bool, AnalyzerError> {
        // 获取select表达式
        let AstNode::SelectExpr(select_left, key) = &mut call.left.node.clone() else {
            unreachable!()
        };

        // 获取left的类型(已经在之前推导过)
        let select_left_type = self.infer_right_expr(select_left, Type::default())?;

        // 解构类型判断
        let select_left_type = if matches!(select_left_type.kind, TypeKind::Ptr(_) | TypeKind::RawPtr(_)) {
            match &select_left_type.kind {
                TypeKind::Ptr(value_type) | TypeKind::RawPtr(value_type) => *value_type.clone(),
                _ => unreachable!(),
            }
        } else {
            select_left_type.clone()
        };

        // 如果是 struct 类型且 key 是其属性,不需要重写
        if let TypeKind::Struct(_, _, properties) = &select_left_type.kind {
            if properties.iter().any(|p| p.key == *key) {
                return Ok(false);
            }
        }

        // 获取impl标识符和参数
        let impl_ident = select_left_type.impl_ident.clone().unwrap();
        let impl_symbol_name = format!("{}_{}", impl_ident, key);
        let impl_args = select_left_type.impl_args.clone();

        // 如果有泛型参数,需要处理hash
        let (final_symbol_name, symbol_id) = if !impl_args.is_empty() {
            // 计算参数hash
            let arg_hash = self.generics_impl_args_hash(&impl_args)?;
            let impl_symbol_name_with_hash = format_generics_ident(impl_symbol_name.clone(), arg_hash);

            // find by hash
            if let Some(symbol_id) = self.symbol_table.find_symbol_id(&impl_symbol_name_with_hash, GLOBAL_SCOPE_ID) {
                (impl_symbol_name_with_hash, Some(symbol_id.clone()))
            } else {
                // 直接通过 impl_symbol_name 查找
                if let Some(symbol_id) = self.symbol_table.find_symbol_id(&impl_symbol_name, GLOBAL_SCOPE_ID) {
                    (impl_symbol_name, Some(symbol_id.clone()))
                } else {
                    return Err(AnalyzerError {
                        start: call.left.start,
                        end: call.left.end,
                        message: format!("type '{}' no impl fn '{}({})':", select_left_type, key, impl_symbol_name),
                    });
                }
            }
        } else {
            // find global scope symbol
            if let Some(symbol_id) = self.symbol_table.find_symbol_id(&impl_symbol_name, GLOBAL_SCOPE_ID) {
                (impl_symbol_name, Some(symbol_id.clone()))
            } else {
                return Err(AnalyzerError {
                    start: call.left.start,
                    end: call.left.end,
                    message: format!("type '{}' no impl fn '{}'", select_left_type, key),
                });
            }
        };

        // 重写 select call 为 ident call, TODO 没有 symbol!?
        call.left = Box::new(Expr::ident(call.left.start, call.left.end, final_symbol_name, symbol_id));

        // 构建新的参数列表
        let mut new_args = Vec::new();

        // 添加self参数
        let mut self_arg = select_left.clone(); // {'a':1}.del('a') -> map_del({'a':1}, 'a')
        if self_arg.type_.is_stack_impl() {
            self_arg.type_ = Type::ptr_of(self_arg.type_.clone());
        }
        new_args.push(self_arg);

        new_args.extend(call.args.iter().cloned());

        call.args = new_args;

        // 设置泛型参数
        call.generics_args = select_left_type.impl_args;

        Ok(true)
    }

    fn generics_args_table(
        &mut self,
        call_data: (Vec<Box<Expr>>, Vec<Type>, bool),
        return_target_type: Type,
        temp_fndef_mutex: Arc<Mutex<AstFnDef>>,
    ) -> Result<HashMap<String, Type>, String> {
        let (mut args, generics_args, spread) = call_data;
        let mut table = HashMap::new();

        let temp_fndef = temp_fndef_mutex.lock().unwrap();

        assert!(temp_fndef.is_generics);

        // 如果存在impl_type,必须提供泛型参数
        if temp_fndef.impl_type.kind.is_exist() {
            assert!(!generics_args.is_empty());
        }

        if generics_args.is_empty() {
            // 保存当前的类型参数栈
            let stash_stack = self.generics_args_stack.clone();
            self.generics_args_stack.clear();

            // 遍历所有参数进行类型推导
            let args_len = args.len();
            for (i, arg) in args.iter_mut().enumerate() {
                let is_spread = spread && (i == args_len - 1);

                // 获取实参类型 (arg 在 infer right expr 中会被修改 type_ 和 target_type 字段)
                let arg_type = self.infer_right_expr(arg, Type::default()).map_err(|e| e.message)?;

                let formal_type = self.select_generics_fn_param(temp_fndef.clone(), i, is_spread);
                if formal_type.err {
                    return Err(format!("too many arguments"));
                }

                // 对形参类型进行还原
                let temp_type = self.reduction_type(formal_type.clone()).map_err(|e| e.message)?;

                // 比较类型并填充泛型参数表
                if !self.type_compare_with_generics(&temp_type, &arg_type, &mut table) {
                    return Err(format!("cannot infer generics type from {} to {}", arg_type, temp_type));
                }
            }

            // 处理返回类型约束
            if !matches!(
                return_target_type.kind,
                TypeKind::Unknown | TypeKind::Void | TypeKind::Union(..) | TypeKind::Null
            ) {
                let temp_type = self.reduction_type(temp_fndef.return_type.clone()).map_err(|e| e.message)?;

                if !self.type_compare_with_generics(&temp_type, &return_target_type, &mut table) {
                    return Err(format!("return type infer failed, expect={}, actual={}", return_target_type, temp_type));
                }
            }

            // 检查所有泛型参数是否都已推导
            if let Some(generics_params) = &temp_fndef.generics_params {
                for param in generics_params {
                    if !table.contains_key(&param.ident) {
                        return Err(format!("generics param {} infer failed", param.ident));
                    }
                }
            }

            // 恢复类型参数栈
            self.generics_args_stack = stash_stack;
        } else {
            // user call 以及提供的泛型参数， 直接使用提供的泛型参数
            if let Some(generics_params) = &temp_fndef.generics_params {
                for (i, arg_type) in generics_args.into_iter().enumerate() {
                    let reduced_type = self.reduction_type(arg_type).map_err(|e| e.message)?;
                    table.insert(generics_params[i].ident.clone(), reduced_type);
                }
            }
        }

        Ok(table)
    }

    fn generics_args_hash(&mut self, generics_params: &Vec<GenericsParam>, args_table: HashMap<String, Type>) -> u64 {
        let mut hash_str = "fn".to_string();

        // 遍历所有泛型参数
        for param in generics_params {
            // 从args_table中获取对应的类型
            let t = args_table.get(&param.ident).unwrap();

            hash_str.push_str(&t.hash().to_string());
        }

        let mut hasher = DefaultHasher::new();
        hash_str.hash(&mut hasher);
        hasher.finish()
    }

    fn generics_special_fn(
        &mut self,
        call_data: (Vec<Box<Expr>>, Vec<Type>, bool),
        target_type: Type,
        temp_fndef_mutex: Arc<Mutex<AstFnDef>>,
    ) -> Result<Arc<Mutex<AstFnDef>>, String> {
        {
            let temp_fndef = temp_fndef_mutex.lock().unwrap();
            assert!(!temp_fndef.is_local);
            assert!(temp_fndef.is_generics);
            assert!(temp_fndef.generics_params.is_some());
        }

        let args_table: HashMap<String, Type> = self.generics_args_table(call_data, target_type, temp_fndef_mutex.clone())?;

        let args_hash = {
            let temp_fndef = temp_fndef_mutex.lock().unwrap();
            self.generics_args_hash(temp_fndef.generics_params.as_ref().unwrap(), args_table.clone())
        };

        let symbol_name = {
            let temp_fndef = temp_fndef_mutex.lock().unwrap();
            //temp_fndef.symbol_name@args_hash
            format_generics_ident(temp_fndef.symbol_name.clone(), args_hash)
        };

        let tpl_fn_mutex = temp_fndef_mutex;

        // 如果当前类型的 special_fn 已经生成，则直接返回即可
        {
            let mut tpl_fn = tpl_fn_mutex.lock().unwrap();
            if tpl_fn.generics_hash_table.is_none() {
                tpl_fn.generics_hash_table = Some(HashMap::new());
            }

            let generics_hash_table = tpl_fn.generics_hash_table.as_ref().unwrap();

            // special fn exists
            if let Some(special_fn) = generics_hash_table.get(&args_hash) {
                return Ok(special_fn.clone());
            }
        }

        // lsp 中无论是 否 singleton 都会 clone 一份, 因为 lsp 会随时新增泛型示例，必须保证 tpl fn 是无污染的
        let special_fn_mutex = {
            let result = GenericSpecialFnClone { global_parent: None }.deep_clone(&tpl_fn_mutex);
            let mut tpl_fn = tpl_fn_mutex.lock().unwrap();

            tpl_fn.generics_hash_table.as_mut().unwrap().insert(args_hash, result.clone());
            result
        };

        {
            let mut special_fn = special_fn_mutex.lock().unwrap();
            special_fn.generics_args_hash = Some(args_hash);
            special_fn.generics_args_table = Some(args_table);
            special_fn.symbol_name = symbol_name;
            assert!(!special_fn.is_local);

            // 注册到全局符号表
            match self.symbol_table.define_symbol(
                special_fn.symbol_name.clone(),
                SymbolKind::Fn(special_fn_mutex.clone()),
                special_fn.symbol_start,
            ) {
                Ok(symbol_id) => {
                    special_fn.symbol_id = Some(symbol_id);
                }
                Err(e) => {
                    panic!("define_symbol failed: {}", e);
                }
            }

            special_fn.type_.status = ReductionStatus::Undo;

            // set type_args_stack
            self.generics_args_stack.push(special_fn.generics_args_table.clone().unwrap());
        }

        self.infer_fn_decl(special_fn_mutex.clone()).map_err(|e| e.message)?;

        self.worklist.push(special_fn_mutex.clone());

        // handle child
        {
            let special_fn = special_fn_mutex.lock().unwrap();
            for child in special_fn.local_children.iter() {
                self.rewrite_local_fndef(child.clone());
                self.infer_fn_decl(child.clone()).map_err(|e| e.message)?;
            }
        }

        self.generics_args_stack.pop();
        Ok(special_fn_mutex)
    }

    fn rewrite_local_fndef(&mut self, fndef_mutex: Arc<Mutex<AstFnDef>>) {
        let mut fndef = fndef_mutex.lock().unwrap();

        // 已经注册并改写完毕，不需要重复改写
        if fndef.generics_args_hash.is_some() {
            return;
        }

        // local fn 直接使用 parent 的 hash
        // 这么做也是为了兼容 generic 的情况
        // 否则 local fn 根本不会存在同名的情况, 另外 local fn 的调用作用域仅仅在当前函数内
        if let Some(global_parent) = &fndef.global_parent {
            let args_hash = global_parent.lock().unwrap().generics_args_hash.unwrap();
            fndef.generics_args_hash = Some(args_hash);
        } else {
            assert!(false);
        }

        // 重写函数名并在符号表中注册
        fndef.symbol_name = format_generics_ident(fndef.symbol_name.clone(), fndef.generics_args_hash.unwrap());

        match self
            .symbol_table
            .define_symbol(fndef.symbol_name.clone(), SymbolKind::Fn(fndef_mutex.clone()), fndef.symbol_start)
        {
            Ok(symbol_id) => {
                fndef.symbol_id = Some(symbol_id);
            }
            Err(e) => {
                panic!("define_symbol failed: {}", e);
            }
        }
    }

    /**
     * 整个 infer 都是在 global scope 中进行的，泛型函数也同样只能在全局函数中生命
     */
    fn infer_generics_special(
        &mut self,
        target_type: Type,
        symbol_id: NodeId,
        call_data: (Vec<Box<Expr>>, Vec<Type>, bool),
    ) -> Result<Option<Arc<Mutex<AstFnDef>>>, String> {
        let symbol = self.symbol_table.get_symbol(symbol_id).unwrap();

        if let SymbolKind::Fn(fndef_mutex) = symbol.kind.clone() {
            {
                let fndef = fndef_mutex.lock().unwrap();
                if fndef.is_local {
                    return Ok(None);
                }

                if !fndef.is_generics {
                    return Ok(None);
                }
            }

            if symbol.defined_in != GLOBAL_SCOPE_ID {
                return Err("generic special fn must be defined in global scope".to_string());
            }

            // 由于存在简单的函数重载，所以需要进行多次批评找到合适的函数， 如果没有重载， fndef 就是目标函数
            let special_fn = self.generics_special_fn(call_data, target_type, fndef_mutex)?;
            return Ok(Some(special_fn));
        }

        return Ok(None);
    }

    pub fn select_generics_fn_param(&mut self, temp_fndef: AstFnDef, index: usize, is_spread: bool) -> Type {
        // let temp_fndef = temp_fndef.lock().unwrap();
        assert!(temp_fndef.is_generics);

        if temp_fndef.rest_param && index >= temp_fndef.params.len() - 1 {
            let last_param_mutex = temp_fndef.params.last().unwrap();
            let last_param_type = last_param_mutex.lock().unwrap().type_.clone();

            if let TypeKind::Vec(element_type) = &last_param_type.kind {
                if is_spread {
                    return last_param_type.clone();
                }

                return *element_type.clone();
            } else {
                assert!(false, "last param type must be vec");
            }
        }

        if index >= temp_fndef.params.len() {
            return Type::error();
        }

        let param = temp_fndef.params[index].lock().unwrap();
        return param.type_.clone();
    }

    pub fn select_fn_param(&mut self, index: usize, target_type_fn: TypeFn, is_spread: bool) -> Type {
        if target_type_fn.rest && index >= target_type_fn.param_types.len() - 1 {
            let last_param_type = target_type_fn.param_types.last().unwrap();

            // 最后一个参数必须是 vec
            assert!(matches!(last_param_type.kind, TypeKind::Vec(..)));

            if is_spread {
                return last_param_type.clone();
            }

            if let TypeKind::Vec(element_type) = &last_param_type.kind {
                return *element_type.clone();
            }
        }

        if index >= target_type_fn.param_types.len() {
            return Type::default();
        }

        return target_type_fn.param_types[index].clone();
    }

    pub fn infer_call_args(&mut self, call: &mut AstCall, target_type_fn: TypeFn) {
        if !target_type_fn.rest && call.args.len() > target_type_fn.param_types.len() {
            self.errors_push(call.left.start, call.left.end, format!("too many args"));
        }

        if !target_type_fn.rest && call.args.len() < target_type_fn.param_types.len() {
            self.errors_push(call.left.start, call.left.end, format!("not enough args"));
        }

        let call_args_len = call.args.len();
        for (i, arg) in call.args.iter_mut().enumerate() {
            let is_spread = call.spread && (i == call_args_len - 1);

            let formal_type = self.select_fn_param(i, target_type_fn.clone(), is_spread);

            if let Err(e) = self.infer_right_expr(arg, formal_type) {
                self.errors_push(e.start, e.end, e.message);
            }
        }
    }

    pub fn infer_call(&mut self, call: &mut AstCall, target_type: Type, start: usize, end: usize) -> Result<Type, AnalyzerError> {
        if call.left.err {
            return Ok(Type::error());
        }

        if matches!(call.left.node, AstNode::SelectExpr(_, _)) {
            let _is_rewrite = self.infer_select_call_rewrite(call)?;
        }
        let call_clone = call.clone();

        if let AstNode::Ident(ident, symbol_id_option) = &mut call.left.node {
            if symbol_id_option.is_none() {
                return Err(AnalyzerError {
                    start: 0,
                    end: 0,
                    message: format!("ident '{}' symbol_id is none", ident),
                });
            }

            let symbol_id = symbol_id_option.unwrap();

            let special_fn = match self.infer_generics_special(
                target_type.clone(),
                symbol_id,
                (call.args.clone(), call.generics_args.clone(), call.spread.clone()),
            ) {
                Ok(result) => result,
                Err(e) => {
                    return Err(AnalyzerError { start, end, message: e });
                }
            };

            match special_fn {
                Some(special_fn) => {
                    let special_fn = special_fn.lock().unwrap();
                    *ident = special_fn.symbol_name.clone();
                    *symbol_id_option = Some(special_fn.symbol_id.unwrap());
                }
                None => {} // local fn 或者 no generics param 都不是 generics fn
            }
        }

        let left_type = self.infer_right_expr(&mut call.left, Type::default())?;

        if !matches!(left_type.kind, TypeKind::Fn(..)) {
            return Err(AnalyzerError {
                start,
                end,
                message: "cannot call non-fn".to_string(),
            });
        }

        let TypeKind::Fn(type_fn) = left_type.kind else { unreachable!() };
        self.infer_call_args(call, *type_fn.clone());
        call.return_type = type_fn.return_type.clone();

        if type_fn.errable {
            // 当前 fn 必须允许 is_errable 或者当前位于 be_caught 中
            let current_fn = self.current_fn_mutex.lock().unwrap();
            if !self.be_caught && !current_fn.is_errable {
                return Err(AnalyzerError {
                    start,
                    end,
                    message: format!(
                        "calling an errable! fn `{}` requires the current `fn {}` errable! as well or be caught.",
                        if type_fn.name.is_empty() { "lambda".to_string() } else { type_fn.name },
                        current_fn.fn_name
                    ),
                });
            }
        }

        return Ok(type_fn.return_type.clone());
    }

    pub fn infer_for_iterator(
        &mut self,
        iterate: &mut Box<Expr>,
        first: &mut Arc<Mutex<VarDeclExpr>>,
        second: &mut Option<Arc<Mutex<VarDeclExpr>>>,
        body: &mut Vec<Box<Stmt>>,
    ) {
        let iterate_type = match self.infer_right_expr(iterate, Type::default()) {
            Ok(iterate_type) => iterate_type,
            Err(e) => {
                self.errors_push(e.start, e.end, e.message);
                return;
            }
        };

        // 检查迭代类型
        match iterate_type.kind {
            TypeKind::Map(..) | TypeKind::Vec(..) | TypeKind::String | TypeKind::Chan(..) => {}
            _ => {
                self.errors_push(
                    iterate.start,
                    iterate.end,
                    format!("for in iterate type must be map/list/string/chan, actual={}", iterate_type),
                );
                return;
            }
        }

        // rewrite var declarations
        self.rewrite_var_decl(first.clone());
        if let Some(second) = second {
            self.rewrite_var_decl(second.clone());
        }

        // 检查 chan 类型的特殊情况
        if matches!(iterate_type.kind, TypeKind::Chan(..)) && second.is_some() {
            self.errors_push(iterate.start, iterate.end, "for chan only have one receive parameter".to_string());
            return;
        }

        // 为变量设置类型
        {
            let mut first_decl = first.lock().unwrap();
            match &iterate_type.kind {
                TypeKind::Map(key_type, _) => {
                    first_decl.type_ = *key_type.clone();
                }
                TypeKind::Chan(chan_type) => {
                    first_decl.type_ = *chan_type.clone();
                }
                TypeKind::String => {
                    if second.is_none() {
                        first_decl.type_ = Type::new(TypeKind::Uint8);
                    } else {
                        first_decl.type_ = Type::new(TypeKind::Int);
                    }
                }
                TypeKind::Vec(element_type) => {
                    if second.is_none() {
                        first_decl.type_ = *element_type.clone();
                    } else {
                        first_decl.type_ = Type::new(TypeKind::Int);
                    }
                }
                _ => unreachable!(),
            }
        }

        // 处理第二个变量的类型
        if let Some(second) = second {
            let mut second_decl = second.lock().unwrap();
            match &iterate_type.kind {
                TypeKind::Map(_, value_type) => {
                    second_decl.type_ = *value_type.clone();
                }
                TypeKind::String => {
                    second_decl.type_ = Type::new(TypeKind::Uint8);
                }
                TypeKind::Vec(element_type) => {
                    second_decl.type_ = *element_type.clone();
                }
                _ => unreachable!(),
            }
        }

        // 处理循环体
        let t = Type::new(TypeKind::Void);
        self.break_target_types.push(t);
        self.infer_body(body);
        self.break_target_types.pop();
    }

    pub fn infer_stmt(&mut self, stmt: &mut Box<Stmt>) -> Result<(), AnalyzerError> {
        match &mut stmt.node {
            AstNode::Fake(expr) => {
                self.infer_right_expr(expr, Type::default())?;
            }
            AstNode::VarDef(var_decl_mutex, expr) => {
                self.infer_vardef(var_decl_mutex, expr)?;
            }
            AstNode::VarTupleDestr(elements, right) => {
                assert!((*elements).len() == elements.len());

                let right_type = self.infer_right_expr(right, Type::default())?;

                if !matches!(right_type.kind, TypeKind::Tuple(..)) {
                    return Err(AnalyzerError {
                        start: right.start,
                        end: right.end,
                        message: format!("cannot assign {} to tuple", right_type),
                    });
                }

                self.infer_var_tuple_destr(elements, right_type, stmt.start, stmt.end)?;
            }
            AstNode::Assign(left, right) => match self.infer_left_expr(left) {
                Ok(left_type) => {
                    if !matches!(left_type.kind, TypeKind::Param(..)) && matches!(left_type.kind, TypeKind::Void) {
                        return Err(AnalyzerError {
                            start: left.start,
                            end: left.end,
                            message: format!("cannot assign to void"),
                        });
                    }

                    self.infer_right_expr(right, left_type)?;
                }
                Err(e) => {
                    return Err(e);
                }
            },
            AstNode::FnDef(_) => {}
            AstNode::Call(call) => {
                self.infer_call(call, Type::new(TypeKind::Void), stmt.start, stmt.end)?;
            }
            AstNode::Catch(try_expr, catch_err_mutex, catch_body) => {
                self.infer_catch(try_expr, catch_err_mutex, catch_body)?;
            }
            AstNode::Select(cases, _has_default, _send_count, _recv_count) => {
                for case in cases.iter_mut() {
                    if let Some(on_call) = &mut case.on_call {
                        if let Err(e) = self.infer_call(on_call, Type::default(), case.start, case.end) {
                            self.errors_push(e.start, e.end, e.message);
                        }
                    }

                    if let Some(recv_var_mutex) = &case.recv_var {
                        {
                            // 存在 recv_var 必定存在 on_call
                            let Some(on_call) = &case.on_call else { unreachable!() };
                            let mut recv_var = recv_var_mutex.lock().unwrap();
                            recv_var.type_ = on_call.return_type.clone();

                            if matches!(recv_var.type_.kind, TypeKind::Unknown | TypeKind::Void | TypeKind::Null) {
                                self.errors_push(
                                    recv_var.symbol_start,
                                    recv_var.symbol_end,
                                    format!("variable declaration cannot use type {}", recv_var.type_),
                                );
                            }
                        }

                        self.rewrite_var_decl(recv_var_mutex.clone());
                    }

                    self.infer_body(&mut case.handle_body);
                }
            }
            AstNode::If(cond, consequent, alternate) => {
                if let Err(e) = self.infer_right_expr(cond, Type::new(TypeKind::Bool)) {
                    self.errors_push(e.start, e.end, e.message);
                }

                self.infer_body(consequent);
                self.infer_body(alternate);
            }
            AstNode::ForCond(condition, body) => {
                if let Err(e) = self.infer_right_expr(condition, Type::new(TypeKind::Bool)) {
                    self.errors_push(e.start, e.end, e.message);
                }

                let t = Type::new(TypeKind::Void);

                self.break_target_types.push(t);
                self.infer_body(body);
                self.break_target_types.pop();
            }
            AstNode::ForIterator(iterate, first, second, body) => {
                self.infer_for_iterator(iterate, first, second, body);
            }
            AstNode::ForTradition(init, condition, update, body) => {
                if let Err(e) = self.infer_stmt(init) {
                    self.errors_push(e.start, e.end, e.message);
                }

                if let Err(e) = self.infer_right_expr(condition, Type::new(TypeKind::Bool)) {
                    self.errors_push(e.start, e.end, e.message);
                }

                if let Err(e) = self.infer_stmt(update) {
                    self.errors_push(e.start, e.end, e.message);
                }

                let t = Type::new(TypeKind::Void);
                self.break_target_types.push(t);
                self.infer_body(body);
                self.break_target_types.pop();
            }
            AstNode::Throw(expr) => {
                {
                    let (is_errable, fn_name, return_type) = {
                        let current_fn = self.current_fn_mutex.lock().unwrap();
                        (current_fn.is_errable, current_fn.fn_name.clone(), current_fn.return_type.clone())
                    };

                    if !is_errable {
                        //  "can't use throw stmt in a fn without an errable! declaration. example: fn %s(...):%s!",
                        self.errors_push(
                            expr.start,
                            expr.end,
                            format!(
                                "can't use throw stmt in a fn without an errable! declaration. example: fn {}(...):{}!",
                                fn_name, return_type
                            ),
                        );
                    }
                }
            }
            AstNode::Return(expr_option) => {
                let target_type = {
                    let current_fn = self.current_fn_mutex.lock().unwrap();
                    current_fn.return_type.clone()
                };

                if let Some(expr) = expr_option {
                    if let Err(e) = self.infer_right_expr(expr, target_type) {
                        self.errors_push(e.start, e.end, e.message);
                    }
                } else {
                    // target_type kind mut void
                    if !matches!(target_type.kind, TypeKind::Void) {
                        self.errors_push(stmt.start, stmt.end, format!("fn expect return type {}, but got void", target_type));
                    }
                }
            }
            AstNode::Break(expr_option) => {
                let target_type = self.break_target_types.last().unwrap().clone();

                // get break target type by break_target_types top
                if let Some(expr) = expr_option {
                    let new_handle_type = self.infer_right_expr(expr, target_type.clone())?;

                    if target_type.kind.is_unknown() {
                        let ptr = self.break_target_types.last_mut().unwrap();
                        *ptr = new_handle_type;
                    }
                } else {
                    // must void
                    if !matches!(target_type.kind, TypeKind::Void) {
                        return Err(AnalyzerError {
                            start: stmt.start,
                            end: stmt.end,
                            message: format!("break missing value expre"),
                        });
                    }
                }
            }
            AstNode::TypeAlias(type_alias_mutex) => {
                self.rewrite_type_alias(type_alias_mutex);
            }
            _ => {}
        }

        Ok(())
    }

    pub fn infer_body(&mut self, body: &mut Vec<Box<Stmt>>) {
        for stmt in body {
            if let Err(e) = self.infer_stmt(stmt) {
                self.errors_push(e.start, e.end, e.message);
            }
        }
    }

    pub fn type_confirm(&mut self, t: &Type) -> bool {
        if t.err {
            return false;
        }

        // 检查基本类型是否为未知类型
        if matches!(t.kind, TypeKind::Unknown) {
            return false;
        }

        // 检查容器类型的元素类型
        match &t.kind {
            // 检查向量类型的元素类型
            TypeKind::Vec(element_type) => {
                if matches!(element_type.kind, TypeKind::Unknown) {
                    return false;
                }
            }

            // 检查映射类型的键值类型
            TypeKind::Map(key_type, value_type) => {
                if matches!(key_type.kind, TypeKind::Unknown) || matches!(value_type.kind, TypeKind::Unknown) {
                    return false;
                }
            }

            // 检查集合类型的元素类型
            TypeKind::Set(element_type) => {
                if matches!(element_type.kind, TypeKind::Unknown) {
                    return false;
                }
            }

            // 检查元组类型的所有元素类型
            TypeKind::Tuple(elements, _) => {
                for element_type in elements {
                    if matches!(element_type.kind, TypeKind::Unknown) {
                        return false;
                    }
                }
            }

            // 其他基本类型都认为是已确认的
            _ => {}
        }

        true
    }

    pub fn type_compare(&mut self, dst: &Type, src: &Type) -> bool {
        self.type_compare_with_generics(dst, src, &mut HashMap::new())
    }

    pub fn type_compare_with_generics(&mut self, dst: &Type, src: &Type, generics_param_table: &mut HashMap<String, Type>) -> bool {
        let mut dst = dst.clone();
        if dst.err || src.err {
            return false;
        }

        // 检查类型状态
        if !matches!(dst.kind, TypeKind::Param(..)) {
            assert!(dst.status == ReductionStatus::Done, "type '{}' not reduction", dst);
        }
        if !matches!(src.kind, TypeKind::Param(..)) {
            assert!(src.status == ReductionStatus::Done, "type '{}' not reduction", src);
        }

        assert!(
            !matches!(dst.kind, TypeKind::Unknown) && !matches!(src.kind, TypeKind::Unknown),
            "type unknown cannot infer"
        );

        // all_t 可以匹配任何类型
        if matches!(dst.kind, TypeKind::AllT) {
            return true;
        }

        // fn_t 可以匹配所有函数类型
        if matches!(dst.kind, TypeKind::FnT) && matches!(src.kind, TypeKind::Fn(..)) {
            return true;
        }

        // raw_ptr<t> 可以赋值为 null 以及 ptr<t>
        if let TypeKind::RawPtr(dst_value_type) = &dst.kind {
            match &src.kind {
                TypeKind::Null => return true,
                TypeKind::Ptr(src_value_type) => {
                    return self.type_compare_with_generics(dst_value_type, src_value_type, generics_param_table);
                }
                _ => {}
            }
        }

        // 处理泛型参数
        if let TypeKind::Param(param_ident) = &dst.kind {
            assert!(src.status == ReductionStatus::Done);
            // let generics_param_table = generics_param_table.as_mut().unwrap();

            if let Some(target_type) = generics_param_table.get(param_ident) {
                dst = target_type.clone();
            } else {
                let target_type = src.clone();
                generics_param_table.insert(param_ident.clone(), target_type.clone());
                dst = target_type;
            }
        }

        // 如果类型不同则返回false
        if dst.kind != src.kind {
            return false;
        }

        // 根据不同类型进行比较
        match (&dst.kind, &src.kind) {
            (TypeKind::Union(left_any, left_types), TypeKind::Union(right_any, right_types)) => {
                if *left_any {
                    return true;
                }
                return self.type_union_compare(&(*left_any, left_types.clone()), &(*right_any, right_types.clone()));
            }

            (TypeKind::Map(left_key, left_value), TypeKind::Map(right_key, right_value)) => {
                if !self.type_compare_with_generics(left_key, right_key, generics_param_table) {
                    return false;
                }
                if !self.type_compare_with_generics(left_value, right_value, generics_param_table) {
                    return false;
                }

                return true;
            }

            (TypeKind::Set(left_element), TypeKind::Set(right_element)) => self.type_compare_with_generics(left_element, right_element, generics_param_table),

            (TypeKind::Chan(left_element), TypeKind::Chan(right_element)) => self.type_compare_with_generics(left_element, right_element, generics_param_table),

            (TypeKind::Vec(left_element), TypeKind::Vec(right_element)) => self.type_compare_with_generics(left_element, right_element, generics_param_table),

            (TypeKind::Arr(left_len, left_element), TypeKind::Arr(right_len, right_element)) => {
                left_len == right_len && self.type_compare_with_generics(left_element, right_element, generics_param_table)
            }

            (TypeKind::Tuple(left_elements, _), TypeKind::Tuple(right_elements, _)) => {
                if left_elements.len() != right_elements.len() {
                    return false;
                }
                left_elements
                    .iter()
                    .zip(right_elements.iter())
                    .all(|(left, right)| self.type_compare_with_generics(left, right, generics_param_table))
            }

            (TypeKind::Fn(left_fn), TypeKind::Fn(right_fn)) => {
                if !self.type_compare_with_generics(&left_fn.return_type, &right_fn.return_type, generics_param_table)
                    || left_fn.param_types.len() != right_fn.param_types.len()
                    || left_fn.rest != right_fn.rest
                    || left_fn.errable != right_fn.errable
                {
                    return false;
                }

                left_fn
                    .param_types
                    .iter()
                    .zip(right_fn.param_types.iter())
                    .all(|(left, right)| self.type_compare_with_generics(left, right, generics_param_table))
            }

            (TypeKind::Struct(_, _, left_props), TypeKind::Struct(_, _, right_props)) => {
                if left_props.len() != right_props.len() {
                    return false;
                }
                left_props
                    .iter()
                    .zip(right_props.iter())
                    .all(|(left, right)| left.key == right.key && self.type_compare_with_generics(&left.type_, &right.type_, generics_param_table))
            }

            (TypeKind::Ptr(left_value), TypeKind::Ptr(right_value)) | (TypeKind::RawPtr(left_value), TypeKind::RawPtr(right_value)) => {
                self.type_compare_with_generics(left_value, right_value, generics_param_table)
            }

            // 其他基本类型直接返回true
            _ => true,
        }
    }

    pub fn infer_fn_decl(&mut self, fndef_mutex: Arc<Mutex<AstFnDef>>) -> Result<Type, AnalyzerError> {
        // 如果已经完成类型推导，直接返回
        let mut fndef = fndef_mutex.lock().unwrap();

        if fndef.type_.status == ReductionStatus::Done {
            return Ok(fndef.type_.clone());
        }

        // 对返回类型进行还原
        match self.reduction_type(fndef.return_type.clone()) {
            Ok(return_type) => {
                fndef.return_type = return_type.clone();
            }
            Err(e) => {
                return Err(e);
            }
        }

        // 处理参数类型
        let mut param_types = Vec::new();

        for (i, param_mutex) in fndef.params.iter().enumerate() {
            let mut param = param_mutex.lock().unwrap();

            // 对参数类型进行还原
            param.type_ = self.reduction_type(param.type_.clone())?;

            // 为什么要在这里进行 ptr of, 只有在 infer 之后才能确定 alias 的具体类型，从而进一步判断是否需要 ptrof
            if fndef.impl_type.kind.is_exist() && i == 0 && param.type_.is_stack_impl() {
                param.type_ = Type::ptr_of(param.type_.clone());
            }

            param_types.push(param.type_.clone());
        }
        let result = Type::new(TypeKind::Fn(Box::new(TypeFn {
            name: fndef.fn_name.clone(),
            tpl: fndef.is_tpl,
            errable: fndef.is_errable,
            rest: fndef.rest_param,
            param_types,
            return_type: fndef.return_type.clone(),
        })));

        fndef.type_ = result.clone();

        Ok(result)
    }

    fn infer_global_vardef(&mut self, var_decl_mutex: &Arc<Mutex<VarDeclExpr>>, right_expr: &mut Box<Expr>) -> Result<(), AnalyzerError> {
        let mut var_decl = var_decl_mutex.lock().unwrap();
        var_decl.type_ = self.reduction_type(var_decl.type_.clone())?;

        let right_expr_type = self.infer_right_expr(right_expr, var_decl.type_.clone())?;

        if var_decl.type_.kind.is_unknown() {
            if !self.type_confirm(&right_expr_type) {
                return Err(AnalyzerError {
                    message: "type infer failed, has type unknown".to_string(),
                    start: right_expr.start,
                    end: right_expr.end,
                });
            }
            var_decl.type_ = right_expr_type;
        }

        Ok(())
    }

    // 新增一个方法来处理模块内的操作
    // fn current_fn_module<F, R>(&mut self, f: F) -> R
    // where
    //     F: FnOnce(&mut Module) -> R,
    // {
    //     let module_index = self.current_fn_mutex.lock().unwrap().module_index;
    //     let mut modules = self.module_db.lock().unwrap();
    //     let module = &mut modules[module_index];
    //     f(module)
    // }

    pub fn rewrite_type_alias(&mut self, type_alias_mutex: &Arc<Mutex<TypeAliasStmt>>) {
        // 如果不存在 params_hash 表示当前 fndef 不存在基于泛型的重写，所以 alias 也不需要进行重写
        if let Some(hash) = self.current_fn_mutex.lock().unwrap().generics_args_hash {
            // alias.ident@hash
            let mut type_alias = type_alias_mutex.lock().unwrap();

            let original_symbol_defined_in = self.symbol_table.get_symbol(type_alias.symbol_id.unwrap()).unwrap().defined_in;

            type_alias.ident = format_generics_ident(type_alias.ident.clone(), hash);

            match self.symbol_table.define_symbol_in_scope(
                type_alias.ident.clone(),
                SymbolKind::TypeAlias(type_alias_mutex.clone()),
                type_alias.symbol_start,
                original_symbol_defined_in,
            ) {
                Ok(symbol_id) => {
                    let original_symbol = self.symbol_table.get_symbol(type_alias.symbol_id.unwrap()).unwrap();
                    original_symbol.generics_id_map.insert(type_alias.ident.clone(), symbol_id);

                    type_alias.symbol_id = Some(symbol_id);
                }
                Err(e) => {
                    panic!("rewrite_type_alias failed: {}", e);
                }
            }
        }
    }

    pub fn rewrite_var_decl(&mut self, var_decl_mutex: Arc<Mutex<VarDeclExpr>>) {
        if let Some(hash) = self.current_fn_mutex.lock().unwrap().generics_args_hash {
            let mut var_decl = var_decl_mutex.lock().unwrap();
            assert!(var_decl.symbol_id.is_some(), "var_decl must have symbol_id");

            // 不能包含 @ 符号
            assert!(!var_decl.ident.contains('@'), "var_decl ident {} cannot contain @ symbol", var_decl.ident);

            // old symbol
            let original_symbol_defined_in = self.symbol_table.get_symbol(var_decl.symbol_id.unwrap()).unwrap().defined_in;

            // ident@arg_hash
            var_decl.ident = format_generics_ident(var_decl.ident.clone(), hash);

            // TODO redefine symbol 需要自定义 scope id, 不然全部定义到 global 中了

            // symbol register
            match self.symbol_table.define_symbol_in_scope(
                var_decl.ident.clone(),
                SymbolKind::Var(var_decl_mutex.clone()),
                var_decl.symbol_start,
                original_symbol_defined_in,
            ) {
                Ok(symbol_id) => {
                    // 基于 old symbol id 获取 symbol 并建立 generics_id_map 映射
                    let original_symbol = self.symbol_table.get_symbol(var_decl.symbol_id.unwrap()).unwrap();
                    original_symbol.generics_id_map.insert(var_decl.ident.clone(), symbol_id);

                    var_decl.symbol_id = Some(symbol_id);
                }
                Err(e) => {
                    panic!("rewrite_var_decl failed: {}", e);
                }
            }
        }
    }

    pub fn infer_var_decl(&mut self, var_decl_mutex: Arc<Mutex<VarDeclExpr>>) -> Result<(), AnalyzerError> {
        let mut var_decl = var_decl_mutex.lock().unwrap();
        var_decl.type_ = self.reduction_type(var_decl.type_.clone())?;

        if matches!(var_decl.type_.kind, TypeKind::Unknown | TypeKind::Void | TypeKind::Null) {
            return Err(AnalyzerError {
                message: format!("variable declaration cannot use type {}", var_decl.type_),
                start: var_decl.symbol_start,
                end: var_decl.symbol_end,
            });
        }

        Ok(())
    }

    pub fn infer_fndef(&mut self, fndef_mutex: Arc<Mutex<AstFnDef>>) {
        self.current_fn_mutex = fndef_mutex;

        let params = {
            let fndef: std::sync::MutexGuard<'_, AstFnDef> = self.current_fn_mutex.lock().unwrap();
            if fndef.type_.status != ReductionStatus::Done {
                return;
            }
            fndef.params.clone()
        };

        // rewrite formal ident
        for var_decl_mutex in params {
            self.rewrite_var_decl(var_decl_mutex);
        }

        // handle body
        {
            // handle body - 修改这部分
            let mut body = {
                let mut fndef = self.current_fn_mutex.lock().unwrap();
                std::mem::take(&mut fndef.body) // 临时取出 body 的所有权
            };

            self.infer_body(&mut body);

            {
                let mut fndef = self.current_fn_mutex.lock().unwrap();
                fndef.body = body;
            }
        }
    }

    fn type_union_compare(&mut self, left: &(bool, Vec<Type>), right: &(bool, Vec<Type>)) -> bool {
        if left.0 == true {
            return true;
        }

        if right.0 && !left.0 {
            return false;
        }

        for right_type in right.1.iter() {
            if left.1.iter().find(|left_type| self.type_compare(left_type, right_type)).is_none() {
                return false;
            }
        }

        return true;
    }

    fn infer_generics_param_constraints(&mut self, impl_type: Type, generics_params: &mut Vec<GenericsParam>) -> Result<(), String> {
        if let TypeKind::Alias(alias) = impl_type.kind {
            let impl_ident = &alias.ident;

            // 从符号表中获取类型别名的定义
            let symbol = self
                .symbol_table
                .find_global_symbol(impl_ident)
                .ok_or_else(|| format!("type alias '{}' not found", impl_ident))?;

            // symbol.kind 为 TypeAlias
            if let SymbolKind::TypeAlias(type_alias_mutex) = symbol.kind.clone() {
                let mut type_alias = type_alias_mutex.lock().unwrap();

                // params.length == generics_params.length
                if type_alias.params.len() != generics_params.len() {
                    return Err(format!("type alias '{}' params length mismatch", impl_ident));
                }

                for (i, type_generics_param) in type_alias.params.iter_mut().enumerate() {
                    if !type_generics_param.constraints.0 {
                        for constraint in &mut type_generics_param.constraints.1 {
                            *constraint = self.reduction_type(constraint.clone()).map_err(|e| e.message)?;
                        }
                    }

                    let impl_generics_param = &generics_params[i];

                    if !self.type_union_compare(&type_generics_param.constraints, &impl_generics_param.constraints) {
                        return Err(format!("type alias '{}' param constraint mismatch", impl_ident));
                    }
                }
            } else {
                return Err(format!("'{}' is not a type alias", impl_ident));
            }
        }

        Ok(())
    }

    /**
     * 返回参数组合 hash 值
     */
    pub fn generics_constraints_product(&mut self, impl_type: Type, generics_params: &mut Vec<GenericsParam>) -> Result<Vec<u64>, String> {
        let mut any_count = 0;
        let generics_params_len = generics_params.len();

        for param in &mut *generics_params {
            if param.constraints.0 {
                any_count += 1;
            } else {
                for element_type in &mut param.constraints.1 {
                    *element_type = self.reduction_type(element_type.clone()).map_err(|e| e.message)?;
                }
            }
        }

        if any_count != 0 && any_count != generics_params_len {
            return Err("all generics params must have constraints or all none".to_string());
        }

        // 如果impl_type是类型别名，需要处理泛型参数约束
        if matches!(impl_type.kind, TypeKind::Alias(..)) {
            self.infer_generics_param_constraints(impl_type, generics_params)?;
        }

        let mut hash_list = Vec::new();

        // 如果所有参数都是any，直接返回空列表
        if any_count == generics_params_len {
            return Ok(hash_list);
        }

        // 生成所有可能的类型组合
        let mut current_product = vec![Type::default(); generics_params_len];
        let mut combinations = Vec::new();

        // 递归生成笛卡尔积
        self.cartesian_product(generics_params, 0, &mut current_product, &mut combinations);

        // 为每个组合计算hash值
        for args_table in combinations {
            let hash = self.generics_args_hash(generics_params, args_table);
            hash_list.push(hash);
        }

        Ok(hash_list)
    }

    // 辅助方法：生成笛卡尔积
    fn cartesian_product(&self, params: &Vec<GenericsParam>, depth: usize, current_product: &mut Vec<Type>, result: &mut Vec<HashMap<String, Type>>) {
        if depth == params.len() {
            // 创建新的参数表
            let mut arg_table = HashMap::new();
            for (i, param) in params.iter().enumerate() {
                arg_table.insert(param.ident.clone(), current_product[i].clone());
            }
            result.push(arg_table);
        } else {
            let param = &params[depth];
            // 遍历当前参数的所有可能类型
            for element_type in &param.constraints.1 {
                assert!(element_type.status == ReductionStatus::Done);
                current_product[depth] = element_type.clone();
                self.cartesian_product(params, depth + 1, current_product, result);
            }
        }
    }

    pub fn pre_infer(&mut self) -> Vec<AnalyzerError> {
        //  - Global variables also contain type information, which needs to be restored and derived
        let mut vardefs = std::mem::take(&mut self.module.global_vardefs);
        for node in &mut vardefs {
            let AstNode::VarDef(var_decl_mutex, right_expr) = node else { unreachable!() };

            if let Err(e) = self.infer_global_vardef(var_decl_mutex, right_expr) {
                self.errors_push(e.start, e.end, e.message);
            }
        }
        self.module.global_vardefs = vardefs;

        // 遍历 module 下的所有的 fndef, 包含 global fn 和 local fn
        let global_fndefs = self.module.global_fndefs.clone();
        for fndef_mutex in global_fndefs {
            let (is_generics, local_children) = {
                let fndef = fndef_mutex.lock().unwrap();
                (fndef.is_generics, fndef.local_children.clone())
            };

            // 不是泛型函数
            if !is_generics {
                if let Err(e) = self.infer_fn_decl(fndef_mutex.clone()) {
                    self.errors_push(e.start, e.end, e.message);
                }

                for child_fndef_mutex in local_children {
                    if let Err(e) = self.infer_fn_decl(child_fndef_mutex) {
                        self.errors_push(e.start, e.end, e.message);
                    }
                }

                continue;
            } else {
                let mut fndef = fndef_mutex.lock().unwrap();
                assert!(fndef.generics_params.is_some());

                match self.generics_constraints_product(fndef.impl_type.clone(), fndef.generics_params.as_mut().unwrap()) {
                    Ok(generics_args) => {
                        if generics_args.len() == 0 {
                            let _ = self
                                .symbol_table
                                .define_symbol(fndef.symbol_name.clone(), SymbolKind::Fn(fndef_mutex.clone()), fndef.symbol_start);
                        } else {
                            for arg_hash in generics_args {
                                // new symbol_name  symbol_name@arg_hash in symbol_table
                                let symbol_name = format!("{}@{}", fndef.symbol_name, arg_hash);
                                let _ = self
                                    .symbol_table
                                    .define_symbol(symbol_name, SymbolKind::Fn(fndef_mutex.clone()), fndef.symbol_start);
                                // TODO generics fn '%s' param constraint redeclared
                            }
                        }
                    }
                    Err(e) => {
                        self.errors_push(fndef.symbol_start, fndef.symbol_end, e);
                    }
                }
            }
        }

        return self.errors.clone();
    }

    pub fn errors_push(&mut self, start: usize, end: usize, message: String) {
        // 判断 current fn 是否属于当前 module
        let current_fn = self.current_fn_mutex.lock().unwrap();
        if current_fn.module_index > 0 && current_fn.module_index != self.module.index {
            return;
        }

        self.errors.push(AnalyzerError { start, end, message });
    }

    pub fn infer(&mut self) -> Vec<AnalyzerError> {
        for fndef_mutex in self.module.global_fndefs.clone() {
            let fndef = fndef_mutex.lock().unwrap();
            // generics fn 不需要进行类型推倒，因为其 param 是不确定的，只需要被调用时由调用方进行 推导
            if fndef.is_generics {
                continue;
            }

            // 经过了 pre_inf fn 的类型必须是确定的，如果不确定则可能是 pre infer 推导类型异常出了什么错误
            if !matches!(fndef.type_.kind, TypeKind::Fn(_)) {
                continue;
            }
            self.worklist.push(fndef_mutex.clone());
        }

        // handle infer worklist, temp data to self
        while let Some(fndef_mutex) = self.worklist.pop() {
            // 先获取需要的数据
            let generics_args_table = {
                let fndef = fndef_mutex.lock().unwrap();
                fndef.generics_args_table.clone()
            };

            if let Some(ref table) = generics_args_table {
                self.generics_args_stack.push(table.clone());
            }

            // clone 只是增加了 arc 的引用计数，内部还是共用的一个锁，所以需要注意死锁问题
            self.infer_fndef(fndef_mutex.clone());

            let fndef = fndef_mutex.lock().unwrap();
            // handle child, 共用 generics arg table
            for child_fndef_mutex in fndef.local_children.clone() {
                self.infer_fndef(child_fndef_mutex);
            }

            if generics_args_table.is_some() {
                self.generics_args_stack.pop();
            }
        }

        return self.errors.clone();
    }
}
