use std::collections::HashMap;
use strum_macros::Display;

#[derive(Debug, Clone)]
pub struct PackageConfig {}

#[derive(Debug, Clone)]
pub struct AnalyzerError {
    pub start: usize,
    pub end: usize,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct Type {
    pub kind: TypeKind,
    pub status: ReductionStatus,
    pub origin_ident: Option<String>, // type alias 原始标识符
    pub origin_type_kind: TypeKind,

    pub impl_ident: Option<String>,
    pub impl_args: Option<Vec<Type>>,
    pub start: usize, // 类型定义开始位置
    pub end: usize,   // 类型定义结束位置
    pub in_heap: bool,
}

impl Default for Type {
    fn default() -> Self {
        Self {
            kind: TypeKind::Unknown,
            status: ReductionStatus::Undo,
            origin_ident: None,
            origin_type_kind: TypeKind::Unknown,
            impl_ident: None,
            impl_args: None,
            start: 0,
            end: 0,
            in_heap: false,
        }
    }
}

impl Type {
    pub fn new(kind: TypeKind) -> Self {
        Self {
            kind: kind.clone(),
            status: ReductionStatus::Done,
            origin_ident: None,
            origin_type_kind: TypeKind::Unknown,
            impl_ident: Some(kind.to_string()),
            impl_args: None,
            start: 0,
            end: 0,
            in_heap: Self::kind_in_heap(&kind),
        }
    }

    pub fn kind_in_heap(kind: &TypeKind) -> bool {
        matches!(
            kind,
            TypeKind::Union(..)
                | TypeKind::String
                | TypeKind::Vec(..)
                | TypeKind::Map(..)
                | TypeKind::Set(..)
                | TypeKind::Tuple(..)
                | TypeKind::GcEnv
                | TypeKind::Fn(..)
                | TypeKind::CoroutineT
                | TypeKind::Chan(..)
        )
    }

    pub fn ptr_of(t: Type) -> Type {
        let ptr_kind = TypeKind::Ptr(Box::new(t.clone()));

        let mut ptr_type = Type::new(ptr_kind);
        ptr_type.in_heap = false;
        ptr_type.origin_ident = None;
        ptr_type.origin_type_kind = TypeKind::Unknown;
        return ptr_type;
    }
}

// type struct property don't concern the value of the property
#[derive(Debug, Clone)]
pub struct TypeStructProperty {
    pub type_: Type,
    pub key: String,
    pub value: Option<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub struct TypeFn {
    pub name: Option<String>,
    pub return_type: Type,
    pub param_types: Vec<Type>,
    pub rest: bool,
    pub tpl: bool,
}

#[derive(Debug, Clone)]
pub struct TypeAlias {
    pub import_as: Option<String>,
    pub ident: String,
    pub args: Option<Vec<Type>>,
}

impl TypeAlias {
    pub fn default() -> Self {
        Self {
            import_as: None,
            ident: String::new(),
            args: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ReductionStatus {
    Undo = 1,
    Doing = 2,
    Done = 3,
}

#[derive(Debug, Clone, Display)]
#[repr(u8)]
pub enum TypeKind {
    #[strum(serialize = "unknown")]
    Unknown,

    // 基础类型
    #[strum(serialize = "null")]
    Null,
    #[strum(serialize = "bool")]
    Bool,

    #[strum(serialize = "i8")]
    Int8,
    #[strum(serialize = "u8")]
    Uint8,
    #[strum(serialize = "i16")]
    Int16,
    #[strum(serialize = "u16")]
    Uint16,
    #[strum(serialize = "i32")]
    Int32,
    #[strum(serialize = "u32")]
    Uint32,
    #[strum(serialize = "i64")]
    Int64,
    #[strum(serialize = "u64")]
    Uint64,
    #[strum(serialize = "int")]
    Int,
    #[strum(serialize = "uint")]
    Uint,

    #[strum(serialize = "f32")]
    Float32,
    #[strum(serialize = "float")]
    Float,
    #[strum(serialize = "f64")]
    Float64,

    // 复合类型
    #[strum(serialize = "string")]
    String,
    #[strum(serialize = "vec")]
    Vec(Box<Type>), // element type

    #[strum(serialize = "arr")]
    Arr(u64, Box<Type>), // (length, element_type)

    #[strum(serialize = "map")]
    Map(Box<Type>, Box<Type>), // (key_type, value_type)

    #[strum(serialize = "set")]
    Set(Box<Type>), // element type

    #[strum(serialize = "tup")]
    Tuple(Vec<Type>, u8), // (elements, align)

    #[strum(serialize = "chan")]
    Chan(Box<Type>), // element type

    #[strum(serialize = "coroutine_t")]
    CoroutineT,

    #[strum(serialize = "struct")]
    Struct(String, u8, Vec<TypeStructProperty>), // (ident, align, properties)

    #[strum(serialize = "fn")]
    Fn(Box<TypeFn>),

    // 指针类型
    #[strum(serialize = "ptr")]
    Ptr(Box<Type>), // value type
    #[strum(serialize = "raw_ptr")]
    RawPtr(Box<Type>), // value type
    #[strum(serialize = "void_ptr")]
    VoidPtr,

    // 编译时特殊临时类型
    #[strum(serialize = "fn_t")]
    FnT,

    #[strum(serialize = "all_t")]
    AllT,

    #[strum(serialize = "void")]
    Void,

    #[strum(serialize = "raw_string")]
    RawString,

    #[strum(serialize = "alias")]
    Alias(Box<TypeAlias>),

    #[strum(serialize = "param")]
    Param(String), // ident

    #[strum(serialize = "union")]
    Union(bool, Vec<Type>), // (any, elements)

    // runtime GC 相关类型
    #[strum(serialize = "gc")]
    Gc,
    #[strum(serialize = "gc_scan")]
    GcScan,
    #[strum(serialize = "gc_noscan")]
    GcNoScan,
    #[strum(serialize = "runtime_fn")]
    GcFn,
    #[strum(serialize = "env")]
    GcEnv,
    #[strum(serialize = "env_value")]
    GcEnvValue,
    #[strum(serialize = "env_values")]
    GcEnvValues,
    #[strum(serialize = "upvalue")]
    GcUpvalue,
}

// ast struct

#[derive(Debug, Clone, PartialEq, Display)]
pub enum ExprOp {
    #[strum(to_string = "none")]
    None,
    #[strum(to_string = "+")]
    Add,
    #[strum(to_string = "-")]
    Sub,
    #[strum(to_string = "*")]
    Mul,
    #[strum(to_string = "/")]
    Div,
    #[strum(to_string = "%")]
    Rem,

    #[strum(to_string = "!")]
    Not,
    #[strum(to_string = "-")]
    Neg,
    #[strum(to_string = "~")]
    Bnot,
    #[strum(to_string = "&")]
    La,
    #[strum(to_string = "&?")]
    SafeLa,
    #[strum(to_string = "&!")]
    UnsafeLa,
    #[strum(to_string = "*")]
    Ia,

    #[strum(to_string = "&")]
    And,
    #[strum(to_string = "|")]
    Or,
    #[strum(to_string = "^")]
    Xor,
    #[strum(to_string = "<<")]
    Lshift,
    #[strum(to_string = ">>")]
    Rshift,

    #[strum(to_string = "<")]
    Lt,
    #[strum(to_string = "<=")]
    Le,
    #[strum(to_string = ">")]
    Gt,
    #[strum(to_string = ">=")]
    Ge,
    #[strum(to_string = "==")]
    Ee,
    #[strum(to_string = "!=")]
    Ne,

    #[strum(to_string = "&&")]
    AndAnd,
    #[strum(to_string = "||")]
    OrOr,
}

#[derive(Debug, Clone)]
pub enum AstNode {
    None,
    Literal(TypeKind, String),            // (kind, value)
    Binary(ExprOp, Box<Expr>, Box<Expr>), // (op, left, right)
    Unary(ExprOp, Box<Expr>),             // (op, operand)
    Ident(String),                        // ident
    As(Type, Box<Expr>),                  // (target_type, src)
    Is(Type, Box<Expr>),                  // (target_type, src)
    MatchIs(Type),                        // (target_type)

    // marco
    MacroSizeof(Type),       // (target_type)
    MacroUla(Box<Expr>),     // (src)
    MacroReflectHash(Type),  // (target_type)
    MacroTypeEq(Type, Type), // (left_type, right_type)
    MacroCoAsync(MacroCoAsyncExpr),
    MacroCall(String, Vec<MacroArg>), // (ident, args)
    MacroDefault,

    New(Type, Vec<StructNewProperty>), // (type_, properties)

    MapAccess(Type, Box<Expr>, Box<Expr>),              // (element_type, left, key)
    VecAccess(Type, Box<Expr>, Box<Expr>),              // (element_type, left, index)
    ArrayAccess(Type, Box<Expr>, Box<Expr>),            // (element_type, left, index)
    TupleAccess(Type, Box<Expr>, u64),                  // (element_type, left, index)
    StructSelect(Box<Expr>, String, StructNewProperty), // (instance, key, property)
    EnvAccess(u8, String),                              // (index, unique_ident)

    VecNew(Vec<Box<Expr>>, Option<Box<Expr>>, Option<Box<Expr>>), // (elements, len, cap)
    ArrayNew(Vec<Box<Expr>>),                                     // elements
    MapNew(Vec<MapElement>),                                      // elements
    SetNew(Vec<Box<Expr>>),                                       //  elements
    TupleNew(Vec<Box<Expr>>),                                     // elements
    TupleDestr(Vec<Box<Expr>>),                                   // elements
    StructNew(String, Type, Vec<StructNewProperty>),              // (ident, type_, properties)
    Try(Box<Expr>, VarDeclExpr, Vec<Box<Stmt>>),                  // (try_expr, catch_err, catch_body)

    // 未推断出具体表达式类型
    EmptyCurlyNew,
    Access(Box<Expr>, Box<Expr>),                // (left, key)
    Select(Box<Expr>, String),                   // (left, key)
    VarDecl(String, Type, bool, Option<String>), // (ident, type_, be_capture, heap_ident)

    // Statements
    Fake(Box<Expr>), // (expr)

    Break(Option<Box<Expr>>), // (expr)
    Continue,
    Import(ImportStmt),                                    // 比较复杂直接保留
    VarDef(VarDeclExpr, Box<Expr>),                        // (var_decl, right)
    VarTupleDestr(Box<TupleDestrExpr>, Box<Expr>),         // (tuple_destr, right)
    Assign(Box<Expr>, Box<Expr>),                          // (left, right)
    Return(Option<Box<Expr>>),                             // (expr)
    If(Box<Expr>, Vec<Box<Stmt>>, Option<Vec<Box<Stmt>>>), // (condition, consequent, alternate)
    Throw(Box<Expr>),
    TryCatch(Box<Expr>, VarDeclExpr, Vec<Box<Stmt>>), // (try_expr, catch_err, catch_body)
    Let(Box<Expr>),                                   // (expr)
    ForIterator(Box<Expr>, VarDeclExpr, Option<VarDeclExpr>, Vec<Box<Stmt>>), // (iterate, first, second, body)

    ForCond(Box<Expr>, Vec<Box<Stmt>>),                            // (condition, body)
    ForTradition(Box<Stmt>, Box<Expr>, Box<Stmt>, Vec<Box<Stmt>>), // (init, cond, update, body)
    TypeAlias(String, Option<Vec<GenericsParam>>, Type),           // (ident, params, type_)

    // 既可以作为表达式，也可以作为语句
    Call(AstCall),
    Catch(Box<Expr>, VarDeclExpr, Vec<Box<Stmt>>), // (try_expr, catch_err, catch_body)
    Match(Option<Box<Expr>>, Vec<MatchCase>),      // (subject, cases)
    FnDef(AstFnDef),
}

impl AstNode {
    pub fn can_assign(&self) -> bool {
        matches!(
            self,
            AstNode::Ident(..)
                | AstNode::Access(..)
                | AstNode::Select(..)
                | AstNode::MapAccess(..)
                | AstNode::VecAccess(..)
                | AstNode::EnvAccess(..)
                | AstNode::StructSelect(..)
        )
    }
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub start: usize,
    pub end: usize,
    pub node: AstNode,
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub start: usize,
    pub end: usize,
    pub type_: Type,
    pub target_type: Type,
    pub node: AstNode,
}

// default
impl Default for Expr {
    fn default() -> Self {
        Self {
            start: 0,
            end: 0,
            type_: Type::default(),
            target_type: Type::default(),
            node: AstNode::None,
        }
    }
}

impl Expr {
    pub fn ident(start: usize, end: usize, literal: String) -> Self {
        Self {
            start,
            end,
            type_: Type::default(),
            target_type: Type::default(),
            node: AstNode::Ident(literal),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VarDeclExpr {
    pub ident: String,
    pub type_: Type,
    pub be_capture: bool,
    pub heap_ident: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AstCall {
    pub return_type: Type,
    pub left: Box<Expr>,
    pub generics_args: Vec<Type>,
    pub args: Vec<Box<Expr>>,
    pub spread: bool,
}

#[derive(Debug, Clone)]
pub struct StructNewProperty {
    pub type_: Type,
    pub key: String,
    pub value: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct MacroCoAsyncExpr {
    pub closure_fn: Box<AstFnDef>,
    pub closure_fn_void: Box<AstFnDef>,
    pub origin_call: Box<AstCall>,
    pub flag_expr: Option<Box<Expr>>,
    pub return_type: Type,
}

#[derive(Debug, Clone)]
pub enum MacroArg {
    Stmt(Box<Stmt>),
    Expr(Box<Expr>),
    Type(Type),
}

#[derive(Debug, Clone)]
pub struct MapElement {
    pub key: Box<Expr>,
    pub value: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct TupleDestrExpr {
    pub elements: Vec<Box<Expr>>,
}

// 语句实现
#[derive(Debug, Clone)]
pub struct ImportStmt {
    pub file: Option<String>,
    pub ast_package: Option<Vec<String>>,
    pub as_name: Option<String>,
    pub module_type: u8,
    pub full_path: String,
    pub package_conf: Option<PackageConfig>,
    pub package_dir: String,
    pub use_links: bool,
    pub module_ident: String,
}

#[derive(Debug, Clone)]
pub struct GenericsParam {
    pub ident: String,
    pub constraints: (bool, Vec<Type>), // (any, elements)
}

impl GenericsParam {
    pub fn new(ident: String) -> Self {
        Self {
            ident,
            constraints: (true, Vec::new()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatchCase {
    pub cond_list: Vec<Box<Expr>>,
    pub is_default: bool,
    pub handle_expr: Option<Box<Expr>>,
    pub handle_body: Option<Vec<Box<Stmt>>>,
}

#[derive(Debug, Clone)]
pub struct AstFnDef {
    pub symbol_name: Option<String>,
    pub closure_name: Option<String>,
    pub return_type: Type,
    pub params: Vec<VarDeclExpr>,
    pub rest_param: bool,
    pub body: Vec<Box<Stmt>>,
    pub closure: Option<isize>,
    pub generics_hash_table: Option<HashMap<String, Type>>,
    pub generics_args_table: Option<HashMap<String, Type>>,
    pub generics_args_hash: Option<String>,
    pub generics_params: Option<Vec<GenericsParam>>,
    pub impl_type: Type,
    pub capture_exprs: Vec<Box<Expr>>,
    pub be_capture_locals: Vec<String>,
    pub type_: Type,
    pub generic_assign: Option<HashMap<String, Type>>,
    pub global_parent: Option<Box<AstFnDef>>,
    pub local_children: Vec<Box<AstFnDef>>,
    pub is_local: bool,
    pub is_tpl: bool,
    pub linkid: Option<String>,
    pub is_generics: bool,
    pub is_co_async: bool,
    pub is_private: bool,
    pub break_target_types: Vec<Type>,
    pub fn_name: Option<String>,
    pub rel_path: Option<String>,

    pub start: usize,
    pub end: usize,
}

// ast fn def default
impl Default for AstFnDef {
    fn default() -> Self {
        Self {
            symbol_name: None,
            closure_name: None,
            return_type: Type::new(TypeKind::Void),
            params: Vec::new(),
            rest_param: false,
            body: Vec::new(),
            closure: None,
            generics_hash_table: None,
            generics_args_table: None,
            generics_args_hash: None,
            generics_params: None,
            impl_type: Type::default(),
            capture_exprs: Vec::new(),
            be_capture_locals: Vec::new(),
            type_: Type::default(),
            generic_assign: None,
            global_parent: None,
            local_children: Vec::new(),
            is_local: false,
            is_tpl: false,
            linkid: None,
            is_generics: false,
            is_co_async: false,
            is_private: false,
            break_target_types: Vec::new(),
            fn_name: None,
            rel_path: None,
            start: 0,
            end: 0,
        }
    }
}
