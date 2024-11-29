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
            TypeKind::Union(_)
                | TypeKind::String
                | TypeKind::Vec(_)
                | TypeKind::Map(_)
                | TypeKind::Set(_)
                | TypeKind::Tuple(_)
                | TypeKind::GcEnv
                | TypeKind::Fn(_)
                | TypeKind::CoroutineT
                | TypeKind::Chan(_)
        )
    }

    pub fn ptr_of(t: Type) -> Type {
        let ptr_kind = TypeKind::Ptr(Box::new(TypePtr(t.clone())));

        let mut ptr_type = Type::new(ptr_kind);
        ptr_type.in_heap = false;
        ptr_type.origin_ident = None;
        ptr_type.origin_type_kind = TypeKind::Unknown;
        return ptr_type;
    }
}

#[derive(Debug, Clone)]
pub struct TypeVec {
    pub element_type: Type,
}

#[derive(Debug, Clone)]
pub struct TypeChan {
    pub element_type: Type,
}

#[derive(Debug, Clone)]
pub struct TypeArray {
    pub length: u64,
    pub element_type: Type,
}

#[derive(Debug, Clone)]
pub struct TypeMap {
    pub key_type: Type,
    pub value_type: Type,
}

#[derive(Debug, Clone)]
pub struct TypeSet {
    pub element_type: Type,
}

#[derive(Debug, Clone)]
pub struct TypeTuple {
    pub elements: Vec<Type>,
    pub align: u8,
}

// type struct property don't concern the value of the property
#[derive(Debug, Clone)]
pub struct TypeStructProperty {
    pub type_: Type,
    pub key: String,
    pub value: Option<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub struct TypeStruct {
    pub ident: String,
    pub align: u8,
    pub properties: Vec<TypeStructProperty>,
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

#[derive(Debug, Clone)]
pub struct TypeParam {
    pub ident: String,
}

#[derive(Debug, Clone)]
pub struct TypePtr(pub Type);

#[derive(Debug, Clone)]
pub struct TypeUnion {
    pub any: bool,
    pub elements: Vec<Type>,
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
    Vec(Box<TypeVec>),
    #[strum(serialize = "arr")]
    Arr(Box<TypeArray>),
    #[strum(serialize = "map")]
    Map(Box<TypeMap>),
    #[strum(serialize = "set")]
    Set(Box<TypeSet>),
    #[strum(serialize = "tup")]
    Tuple(Box<TypeTuple>),

    #[strum(serialize = "chan")]
    Chan(Box<TypeChan>),

    #[strum(serialize = "coroutine_t")]
    CoroutineT,

    #[strum(serialize = "struct")]
    Struct(Box<TypeStruct>),

    #[strum(serialize = "fn")]
    Fn(Box<TypeFn>),

    // 指针类型
    #[strum(serialize = "ptr")]
    Ptr(Box<TypePtr>),
    #[strum(serialize = "raw_ptr")]
    RawPtr(Box<TypePtr>),
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
    Param(Box<TypeParam>),
    #[strum(serialize = "union")]
    Union(Box<TypeUnion>),

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
    Literal(LiteralExpr),
    Binary(BinaryExpr),
    Unary(UnaryExpr),
    Ident(IdentExpr),
    As(AsExpr),
    Is(IsExpr),
    MatchIs(MatchIsExpr),

    // marco
    MacroSizeof(MacroSizeofExpr),
    MacroUla(MacroUlaExpr),
    MacroReflectHash(MacroReflectHashExpr),
    MacroTypeEq(MacroTypeEqExpr),
    MacroCoAsync(MacroCoAsyncExpr),
    MacroCall(MacroCallExpr),
    MacroDefault(MacroDefaultExpr),

    New(NewExpr),

    MapAccess(MapAccessExpr),
    VecAccess(VecAccessExpr),
    ArrayAccess(ArrayAccessExpr),
    TupleAccess(TupleAccessExpr),
    StructSelect(StructSelectExpr),
    EnvAccess(EnvAccessExpr),

    VecNew(VecNewExpr),
    ArrayNew(ArrayNewExpr),
    MapNew(MapNewExpr),
    SetNew(SetNewExpr),
    TupleNew(TupleNewExpr),
    TupleDestr(TupleDestrExpr),
    StructNew(StructNewExpr),
    Try(TryExpr),

    // 未推断出具体表达式类型
    EmptyCurlyNew(EmptyCurlyNewExpr),
    Access(AccessExpr),
    Select(SelectExpr),
    VarDecl(VarDeclExpr), // 抽象复合类型，用于 fn 的参数或者 for 的 k,v

    // Statements
    Fake(FakeStmt),

    Break(BreakStmt),
    Continue(ContinueStmt),
    Import(ImportStmt),
    VarDef(VarDefStmt),
    VarTupleDestr(VarTupleDestrStmt),
    Assign(AssignStmt),
    Return(ReturnStmt),
    If(IfStmt),
    Throw(ThrowStmt),
    TryCatch(TryCatchStmt),
    Let(LetStmt),
    ForIterator(ForIteratorStmt),
    ForCond(ForCondStmt),
    ForTradition(ForTraditionStmt),
    TypeAlias(TypeAliasStmt),

    // 既可以作为表达式，也可以作为语句
    Call(AstCall),
    Catch(AstCatch),
    Match(AstMatch),
    FnDef(AstFnDef),
}

impl AstNode {
    pub fn can_assign(&self) -> bool {
        matches!(self, AstNode::Ident(_) | AstNode::Access(_) | AstNode::Select(_) | AstNode::MapAccess(_) | AstNode::VecAccess(_) | AstNode::EnvAccess(_) | AstNode::StructSelect(_))
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
            node: AstNode::Ident(IdentExpr { literal }),
        }
    }
}


#[derive(Debug, Clone)]
pub struct FakeStmt {
    pub expr: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct ContinueStmt;

#[derive(Debug, Clone)]
pub struct BreakStmt {
    pub expr: Option<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub struct IdentExpr {
    pub literal: String,
}

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub operator: ExprOp,
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct UnaryExpr {
    pub operator: ExprOp,
    pub operand: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct LiteralExpr {
    pub kind: TypeKind,
    pub value: String,
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

// var a = new Type
#[derive(Debug, Clone)]
pub struct NewExpr {
    pub type_: Type,
    pub properties: Vec<StructNewProperty>,
}

#[derive(Debug, Clone)]
pub struct AsExpr {
    pub target_type: Type,
    pub src: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct IsExpr {
    pub target_type: Type,
    pub src: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct MatchIsExpr {
    pub target_type: Type,
}

#[derive(Debug, Clone)]
pub struct MacroDefaultExpr;

#[derive(Debug, Clone)]
pub struct MacroSizeofExpr {
    pub target_type: Type,
}

#[derive(Debug, Clone)]
pub struct MacroUlaExpr {
    pub src: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct MacroTypeEqExpr {
    pub left_type: Type,
    pub right_type: Type,
}

#[derive(Debug, Clone)]
pub struct MacroReflectHashExpr {
    pub target_type: Type,
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
pub struct MacroCallExpr {
    pub ident: String,
    pub args: Vec<MacroArg>,
}

#[derive(Debug, Clone)]
pub struct AccessExpr {
    pub left: Box<Expr>,
    pub key: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct MapAccessExpr {
    pub element_type: Type,
    pub left: Box<Expr>,
    pub key: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct VecAccessExpr {
    pub element_type: Type,
    pub left: Box<Expr>,
    pub index: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct ArrayAccessExpr {
    pub element_type: Type,
    pub left: Box<Expr>,
    pub index: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct TupleAccessExpr {
    pub element_type: Type,
    pub left: Box<Expr>,
    pub index: u64,
}

#[derive(Debug, Clone)]
pub struct EnvAccessExpr {
    pub index: u8,
    pub unique_ident: String,
}

#[derive(Debug, Clone)]
pub struct SelectExpr {
    pub left: Box<Expr>,
    pub key: String,
}

#[derive(Debug, Clone)]
pub struct StructSelectExpr {
    pub instance: Box<Expr>,
    pub key: String,
    pub property: StructNewProperty,
}

#[derive(Debug, Clone)]
pub struct VecNewExpr {
    pub elements: Vec<Box<Expr>>,
    pub len: Option<Box<Expr>>,
    pub cap: Option<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub struct ArrayNewExpr {
    pub elements: Vec<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub struct MapElement {
    pub key: Box<Expr>,
    pub value: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct MapNewExpr {
    pub elements: Vec<MapElement>,
}

#[derive(Debug, Clone)]
pub struct SetNewExpr {
    pub elements: Vec<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub struct EmptyCurlyNewExpr;

#[derive(Debug, Clone)]
pub struct TupleNewExpr {
    pub elements: Vec<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub struct TupleDestrExpr {
    pub elements: Vec<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub struct StructNewExpr {
    pub ident: String,
    pub type_: Type,
    pub properties: Vec<StructNewProperty>,
}

#[derive(Debug, Clone)]
pub struct TryExpr {
    pub try_expr: Box<Expr>,
    pub catch_err: VarDeclExpr,
    pub catch_body: Vec<Box<Stmt>>,
}

#[derive(Debug, Clone)]
pub struct BoomExpr {
    pub expr: Box<Expr>,
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
pub struct VarDefStmt {
    pub var_decl: VarDeclExpr,
    pub right: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct VarTupleDestrStmt {
    pub tuple_destr: Box<TupleDestrExpr>,
    pub right: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct AssignStmt {
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct ReturnStmt {
    pub expr: Option<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub struct IfStmt {
    pub condition: Box<Expr>,
    pub consequent: Vec<Box<Stmt>>,
    pub alternate: Option<Vec<Box<Stmt>>>,
}

#[derive(Debug, Clone)]
pub struct ThrowStmt {
    pub error: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct TryCatchStmt {
    pub try_body: Vec<Box<Stmt>>,
    pub catch_err: VarDeclExpr,
    pub catch_handle: Vec<Box<Stmt>>,
}

#[derive(Debug, Clone)]
pub struct LetStmt {
    pub expr: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct ForIteratorStmt {
    pub iterate: Box<Expr>,
    pub first: VarDeclExpr,
    pub second: Option<VarDeclExpr>,
    pub body: Vec<Box<Stmt>>,
}

#[derive(Debug, Clone)]
pub struct ForCondStmt {
    pub condition: Box<Expr>,
    pub body: Vec<Box<Stmt>>,
}

#[derive(Debug, Clone)]
pub struct ForTraditionStmt {
    pub init: Box<Stmt>,
    pub cond: Box<Expr>,
    pub update: Box<Stmt>,
    pub body: Vec<Box<Stmt>>,
}

#[derive(Debug, Clone)]
pub struct TypeAliasStmt {
    pub ident: String,
    pub params: Option<Vec<GenericsParam>>,
    pub type_: Type,
}

#[derive(Debug, Clone)]
pub struct GenericsParam {
    pub ident: String,
    pub constraints: TypeUnion,
}

impl GenericsParam {
    pub fn new(ident: String) -> Self {
        Self {
            ident,
            constraints: TypeUnion {
                any: true,
                elements: Vec::new(),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct AstCatch {
    pub try_expr: Box<Expr>,
    pub catch_err: VarDeclExpr,
    pub catch_body: Vec<Box<Stmt>>,
}

#[derive(Debug, Clone)]
pub struct MatchCase {
    pub cond_list: Vec<Box<Expr>>,
    pub is_default: bool,
    pub handle_expr: Option<Box<Expr>>,
    pub handle_body: Option<Vec<Box<Stmt>>>,
}

#[derive(Debug, Clone)]
pub struct AstMatch {
    pub subject: Option<Box<Expr>>,
    pub cases: Vec<MatchCase>,
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

// 辅助函数实现
impl IdentExpr {
    pub fn new(literal: String) -> Self {
        Self { literal }
    }
}
