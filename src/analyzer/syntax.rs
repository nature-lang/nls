use super::common::*;
use super::lexer::semantic_token_type_index;
use super::lexer::Token;
use super::lexer::TokenType;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};
use tower_lsp::lsp_types::SemanticTokenType;

const LOCAL_FN_NAME: &str = "lambda";

pub struct SyntaxError(usize, usize, String);

impl fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SyntaxError: {}", self.2)
    }
}

impl fmt::Debug for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SyntaxError: {}", self.2)
    }
}
impl Error for SyntaxError {}

pub fn token_to_expr_op(token: &TokenType) -> ExprOp {
    match token {
        TokenType::Plus => ExprOp::Add,
        TokenType::Minus => ExprOp::Sub,
        TokenType::Star => ExprOp::Mul,
        TokenType::Slash => ExprOp::Div,
        TokenType::Percent => ExprOp::Rem,
        TokenType::EqualEqual => ExprOp::Ee,
        TokenType::NotEqual => ExprOp::Ne,
        TokenType::GreaterEqual => ExprOp::Ge,
        TokenType::RightAngle => ExprOp::Gt,
        TokenType::LessEqual => ExprOp::Le,
        TokenType::LessThan => ExprOp::Lt,
        TokenType::AndAnd => ExprOp::AndAnd,
        TokenType::OrOr => ExprOp::OrOr,

        // 位运算
        TokenType::Tilde => ExprOp::Bnot,
        TokenType::And => ExprOp::And,
        TokenType::Or => ExprOp::Or,
        TokenType::Xor => ExprOp::Xor,
        TokenType::LeftShift => ExprOp::Lshift,
        TokenType::RightShift => ExprOp::Rshift,

        // equal 快捷运算拆解
        TokenType::PercentEqual => ExprOp::Rem,
        TokenType::MinusEqual => ExprOp::Sub,
        TokenType::PlusEqual => ExprOp::Add,
        TokenType::SlashEqual => ExprOp::Div,
        TokenType::StarEqual => ExprOp::Mul,
        TokenType::OrEqual => ExprOp::Or,
        TokenType::AndEqual => ExprOp::And,
        TokenType::XorEqual => ExprOp::Xor,
        TokenType::LeftShiftEqual => ExprOp::Lshift,
        TokenType::RightShiftEqual => ExprOp::Rshift,
        _ => ExprOp::None,
    }
}

pub fn token_to_type_kind(token: &TokenType) -> TypeKind {
    match token {
        // literal
        TokenType::True | TokenType::False => TypeKind::Bool,
        TokenType::Null => TypeKind::Null,
        TokenType::Void => TypeKind::Void,
        TokenType::FloatLiteral => TypeKind::Float,
        TokenType::IntLiteral => TypeKind::Int,
        TokenType::StringLiteral => TypeKind::String,

        // type
        TokenType::Bool => TypeKind::Bool,
        TokenType::Float => TypeKind::Float,
        TokenType::F32 => TypeKind::Float32,
        TokenType::F64 => TypeKind::Float64,
        TokenType::Int => TypeKind::Int,
        TokenType::I8 => TypeKind::Int8,
        TokenType::I16 => TypeKind::Int16,
        TokenType::I32 => TypeKind::Int32,
        TokenType::I64 => TypeKind::Int64,
        TokenType::Uint => TypeKind::Uint,
        TokenType::U8 => TypeKind::Uint8,
        TokenType::U16 => TypeKind::Uint16,
        TokenType::U32 => TypeKind::Uint32,
        TokenType::U64 => TypeKind::Uint64,
        TokenType::String => TypeKind::String,
        TokenType::Var => TypeKind::Unknown,
        _ => TypeKind::Unknown,
    }
}

#[derive(Clone, Copy)]
struct ParserRule {
    prefix: Option<fn(&mut Syntax) -> Result<Box<Expr>, SyntaxError>>,
    infix: Option<fn(&mut Syntax, Box<Expr>) -> Result<Box<Expr>, SyntaxError>>,
    infix_precedence: SyntaxPrecedence,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(u8)] // 指定底层表示类型
pub enum SyntaxPrecedence {
    Null, // 最低优先级
    Assign,
    Catch,
    OrOr,     // ||
    AndAnd,   // &&
    Or,       // |
    Xor,      // ^
    And,      // &
    CmpEqual, // == !=
    Compare,  // > < >= <=
    Shift,    // << >>
    Term,     // + -
    Factor,   // * / %
    TypeCast, // as/is
    Unary,    // - ! ~ * &
    Call,     // foo.bar foo["bar"] foo() foo().foo.bar
    Primary,  // 最高优先级
}

impl SyntaxPrecedence {
    fn next(self) -> Option<Self> {
        let next_value = (self as u8).checked_add(1)?;
        if next_value <= (Self::Primary as u8) {
            // 使用 unsafe 是安全的,因为我们已经确保值在枚举范围内
            Some(unsafe { std::mem::transmute(next_value) })
        } else {
            None
        }
    }
}

pub struct Syntax {
    token_db: Vec<Token>,
    token_indexes: Vec<usize>,
    current: usize, // token index

    lambda_index: usize, // default 0
    errors: Vec<AnalyzerError>,

    // parser 阶段辅助记录当前的 type_param, 当进入到 fn body 或者 struct def 时可以准确识别当前是 type param 还是 alias, 仅仅使用到 key
    // 默认是一个空 hashmap
    type_params_table: HashMap<String, String>,

    // 部分表达式只有在 match cond 中可以使用，比如 is T, n if n xxx 等, parser_match_cond 为 true 时，表示当前处于 match cond 中
    match_cond: bool,

    // match 表达式中 subject 的解析
    match_subject: bool,
}

impl Syntax {
    // static method new, Syntax::new(tokens)
    pub fn new(token_db: Vec<Token>, token_indexes: Vec<usize>) -> Self {
        Self {
            token_db: token_db,
            token_indexes: token_indexes,
            current: 0,
            type_params_table: HashMap::new(),
            match_cond: false,
            match_subject: false,
            errors: Vec::new(),
            lambda_index: 0,
        }
    }

    fn set_current_token_type(&mut self, token_type: SemanticTokenType) {
        if self.current >= self.token_indexes.len() {
            panic!("syntax::peek: current index out of range");
        }

        self.token_db[self.token_indexes[self.current]].semantic_token_type = semantic_token_type_index(token_type);
    }

    fn advance(&mut self) -> &Token {
        assert!(self.current + 1 < self.token_indexes.len(), "Syntax::advance: current index out of range");

        let token_index = self.token_indexes[self.current];

        self.current += 1;
        return &self.token_db[token_index];
    }

    // 当前符号不是 EOF 时才能 advance
    fn safe_advance(&mut self) -> Result<&Token, SyntaxError> {
        let token_index = self.token_indexes[self.current];

        if self.token_db[token_index].token_type == TokenType::Eof {
            return Err(SyntaxError(self.peek().start, self.peek().end, "unexpected end of file".to_string()));
        }

        self.current += 1;
        return Ok(&self.token_db[token_index]);
    }

    fn peek(&self) -> &Token {
        if self.current >= self.token_indexes.len() {
            panic!("syntax::peek: current index out of range");
        }
        return &self.token_db[self.token_indexes[self.current]];
    }

    fn peek_mut(&mut self) -> &mut Token {
        if self.current >= self.token_indexes.len() {
            panic!("syntax::peek: current index out of range");
        }
        return &mut self.token_db[self.token_indexes[self.current]];
    }

    fn prev(&self) -> Option<&Token> {
        if self.current == 0 {
            return None;
        }

        return Some(&self.token_db[self.token_indexes[self.current - 1]]);
    }

    fn is(&self, token_type: TokenType) -> bool {
        return self.peek().token_type == token_type;
    }

    fn consume(&mut self, token_type: TokenType) -> bool {
        if self.is(token_type) {
            self.advance();
            return true;
        }
        return false;
    }

    fn must(&mut self, expect: TokenType) -> Result<&Token, SyntaxError> {
        let token = self.peek().clone(); // 对 self 进行了不可变借用, clone 让借用立刻结束

        if token.token_type != expect {
            let message = format!("expected '{}'", expect.to_string());
            return Err(SyntaxError(token.start, token.end, message));
        }

        if self.current + 1 >= self.token_indexes.len() {
            return Err(SyntaxError(token.start, token.end, "unexpected end of file".to_string()));
        }

        self.advance();

        return Ok(self.prev().unwrap());
    }

    // 对应 parser_next
    fn next(&self, step: usize) -> Option<&Token> {
        if self.current + step >= self.token_indexes.len() {
            return None;
        }
        Some(&self.token_db[self.token_indexes[self.current + step]])
    }

    // 对应 parser_next_is
    fn next_is(&self, step: usize, expect: TokenType) -> bool {
        match self.next(step) {
            Some(token) => token.token_type == expect,
            None => false,
        }
    }

    fn is_stmt_eof(&self) -> bool {
        self.is(TokenType::StmtEof) || self.is(TokenType::Eof)
    }

    fn stmt_new(&self) -> Box<Stmt> {
        Box::new(Stmt {
            start: self.peek().start,
            end: self.peek().end,
            node: AstNode::None,
        })
    }

    fn expr_new(&self) -> Box<Expr> {
        Box::new(Expr {
            start: self.peek().start,
            end: self.peek().end,
            type_: Type::default(),
            target_type: Type::default(),
            node: AstNode::None,
            err: false,
        })
    }

    fn fake_new(&self, expr: Box<Expr>) -> Box<Stmt> {
        let mut stmt = self.stmt_new();
        stmt.node = AstNode::Fake(expr);
        stmt.end = self.prev().unwrap().end;

        return stmt;
    }

    fn find_rule(&self, token_type: TokenType) -> ParserRule {
        use TokenType::*;
        match token_type {
            LeftParen => ParserRule {
                prefix: Some(Self::parser_left_paren_expr),
                infix: Some(Self::parser_call_expr),
                infix_precedence: SyntaxPrecedence::Call,
            },
            LeftSquare => ParserRule {
                prefix: Some(Self::parser_vec_new),
                infix: Some(Self::parser_access),
                infix_precedence: SyntaxPrecedence::Call,
            },
            LeftCurly => ParserRule {
                prefix: Some(Self::parser_left_curly_expr),
                infix: None,
                infix_precedence: SyntaxPrecedence::Null,
            },
            LessThan => ParserRule {
                prefix: None,
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::Compare,
            },
            LeftAngle => ParserRule {
                prefix: None,
                infix: Some(Self::parser_type_args_expr),
                infix_precedence: SyntaxPrecedence::Call,
            },
            MacroIdent => ParserRule {
                prefix: Some(Self::parser_macro_call),
                infix: None,
                infix_precedence: SyntaxPrecedence::Null,
            },
            Dot => ParserRule {
                prefix: None,
                infix: Some(Self::parser_select_expr),
                infix_precedence: SyntaxPrecedence::Call,
            },
            Minus => ParserRule {
                prefix: Some(Self::parser_unary),
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::Term,
            },
            Plus => ParserRule {
                prefix: None,
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::Term,
            },
            Not => ParserRule {
                prefix: Some(Self::parser_unary),
                infix: None,
                infix_precedence: SyntaxPrecedence::Unary,
            },
            Tilde => ParserRule {
                prefix: Some(Self::parser_unary),
                infix: None,
                infix_precedence: SyntaxPrecedence::Unary,
            },
            And => ParserRule {
                prefix: Some(Self::parser_unary),
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::And,
            },
            Or => ParserRule {
                prefix: None,
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::Or,
            },
            Xor => ParserRule {
                prefix: None,
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::Xor,
            },
            LeftShift => ParserRule {
                prefix: None,
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::Shift,
            },
            Percent => ParserRule {
                prefix: None,
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::Factor,
            },
            Star => ParserRule {
                prefix: Some(Self::parser_unary),
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::Factor,
            },
            Slash => ParserRule {
                prefix: None,
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::Factor,
            },
            OrOr => ParserRule {
                prefix: None,
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::OrOr,
            },
            AndAnd => ParserRule {
                prefix: None,
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::AndAnd,
            },
            NotEqual | EqualEqual => ParserRule {
                prefix: None,
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::CmpEqual,
            },

            RightShift => ParserRule {
                prefix: None,
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::Shift,
            },

            RightAngle | GreaterEqual | LessEqual => ParserRule {
                prefix: None,
                infix: Some(Self::parser_binary),
                infix_precedence: SyntaxPrecedence::Compare,
            },
            StringLiteral | IntLiteral | FloatLiteral | True | False | Null => ParserRule {
                prefix: Some(Self::parser_literal),
                infix: None,
                infix_precedence: SyntaxPrecedence::Null,
            },
            As => ParserRule {
                prefix: None,
                infix: Some(Self::parser_as_expr),
                infix_precedence: SyntaxPrecedence::TypeCast,
            },
            Is => ParserRule {
                prefix: Some(Self::parser_match_is_expr),
                infix: Some(Self::parser_is_expr),
                infix_precedence: SyntaxPrecedence::TypeCast,
            },
            Catch => ParserRule {
                prefix: None,
                infix: Some(Self::parser_catch_expr),
                infix_precedence: SyntaxPrecedence::Catch,
            },
            Ident => ParserRule {
                prefix: Some(Self::parser_ident_expr),
                infix: None,
                infix_precedence: SyntaxPrecedence::Null,
            },
            _ => ParserRule {
                prefix: None,
                infix: None,
                infix_precedence: SyntaxPrecedence::Null,
            },
        }
    }

    // 处理中缀表达式的 token
    fn parser_infix_token(&mut self, expr: &Box<Expr>) -> TokenType {
        let mut infix_token = self.peek().token_type.clone();

        // 处理 < 的歧义
        if infix_token == TokenType::LeftAngle && !self.parser_left_angle_is_type_args(expr) {
            let token = self.peek_mut();
            token.token_type = TokenType::LessThan;
            infix_token = TokenType::LessThan;
        }

        // 处理连续的 >> 合并
        if infix_token == TokenType::RightAngle && self.next_is(1, TokenType::RightAngle) {
            self.advance();

            let token = self.peek_mut();
            token.token_type = TokenType::RightShift;
            infix_token = TokenType::RightShift;
        }

        infix_token
    }

    fn must_stmt_end(&mut self) -> Result<(), SyntaxError> {
        if self.is(TokenType::Eof) || self.is(TokenType::RightCurly) {
            return Ok(());
        }

        // ; (scanner 时主动添加)
        if self.is(TokenType::StmtEof) {
            self.advance();
            return Ok(());
        }

        let prev_token = self.prev().unwrap();
        // stmt eof 失败。报告错误，并返回 false 即可
        // 获取前一个 token 的位置用于错误报告
        return Err(SyntaxError(
            prev_token.start,
            prev_token.end,
            "expected ';' or '}' at end of statement".to_string(),
        ));
    }

    fn is_basic_type(&self) -> bool {
        matches!(
            self.peek().token_type,
            TokenType::Var
                | TokenType::Null
                | TokenType::Void
                | TokenType::Int
                | TokenType::I8
                | TokenType::I16
                | TokenType::I32
                | TokenType::I64
                | TokenType::Uint
                | TokenType::U8
                | TokenType::U16
                | TokenType::U32
                | TokenType::U64
                | TokenType::Float
                | TokenType::F32
                | TokenType::F64
                | TokenType::Bool
                | TokenType::String
        )
    }

    pub fn parser(&mut self) -> (Vec<Box<Stmt>>, Vec<Token>, Vec<AnalyzerError>) {
        self.current = 0;

        let mut stmt_list = Vec::new();

        while !self.is(TokenType::Eof) {
            match self.parser_global_stmt() {
                Ok(stmt) => stmt_list.push(stmt),
                Err(e) => {
                    self.errors.push(AnalyzerError {
                        start: e.0,
                        end: e.1,
                        message: e.2,
                    });

                    // 查找到下一个同步点
                    let found = self.synchronize(0);
                    if !found && !self.is(TokenType::Eof) {
                        // 当前字符无法被表达式解析，且 sync 查找下一个可用同步点失败，直接跳过当前字符
                        self.advance();
                    }
                }
            }
        }

        return (stmt_list, self.token_db.clone(), self.errors.clone());
    }

    fn parser_body(&mut self) -> Result<Vec<Box<Stmt>>, SyntaxError> {
        let mut stmt_list = Vec::new();
        self.must(TokenType::LeftCurly)?;

        while !self.is(TokenType::RightCurly) {
            if self.is(TokenType::Eof) {
                return Err(SyntaxError(
                    self.prev().unwrap().start,
                    self.prev().unwrap().end,
                    "unexpected end of file, expected '}'".to_string(),
                ));
            }

            match self.parser_stmt() {
                Ok(stmt) => stmt_list.push(stmt),
                Err(e) => {
                    self.errors.push(AnalyzerError {
                        start: e.0,
                        end: e.1,
                        message: e.2,
                    });

                    let found = self.synchronize(1);
                    if !found && !self.is(TokenType::Eof) {
                        self.advance();
                    }
                }
            }
        }
        self.must(TokenType::RightCurly)?;

        return Ok(stmt_list);
    }

    fn synchronize(&mut self, current_brace_level: isize) -> bool {
        let mut brace_level = current_brace_level;

        loop {
            let token: TokenType = self.peek().token_type.clone();

            // 提前返回的情况
            match token {
                TokenType::Eof => return false,

                // 在当前层级遇到语句结束符
                TokenType::StmtEof if brace_level == current_brace_level => {
                    self.advance();
                    return true;
                }

                // 在当前层级遇到关键字或基本类型
                _ if brace_level == current_brace_level => {
                    // brace_level = 0 时只识别全局级别的语句
                    if brace_level == 0 {
                        if matches!(token, TokenType::Fn | TokenType::Var | TokenType::Import | TokenType::Type) || self.is_basic_type() {
                            return true;
                        }
                    } else {
                        // brace_level > 0 时可以识别函数体内的所有语句类型
                        if matches!(
                            token,
                            TokenType::Var
                                | TokenType::Return
                                | TokenType::If
                                | TokenType::For
                                | TokenType::Match
                                | TokenType::Try
                                | TokenType::Catch
                                | TokenType::Continue
                                | TokenType::Break
                                | TokenType::Import
                                | TokenType::Type
                        ) {
                            return true;
                        }
                    }
                }
                _ => {}
            }

            // 处理花括号层级
            match token {
                TokenType::LeftCurly => brace_level += 1,
                TokenType::RightCurly => {
                    brace_level -= 1;
                    if brace_level < current_brace_level {
                        // 如果小于当前层级，则说明查找同步点失败。
                        return false;
                    }
                }
                _ => {}
            }

            self.advance();
            if self.is(TokenType::Eof) {
                return false;
            }
        }
    }

    fn parser_single_type(&mut self) -> Result<Type, SyntaxError> {
        let mut t = Type::default();
        t.status = ReductionStatus::Undo;
        t.start = self.peek().start;

        // any union type
        if self.consume(TokenType::Any) {
            t.kind = TypeKind::Union(true, Vec::new());
            t.end = self.prev().unwrap().end;
            return Ok(t);
        }

        // 基本类型 int/float/bool/string/void/var
        if self.is_basic_type() {
            // 已经明确了 current 是 basic type token, 才能使用 advance
            let type_token = self.advance();
            t.kind = token_to_type_kind(&type_token.token_type);
            t.impl_ident = Some(t.kind.to_string());

            if matches!(type_token.token_type, TokenType::Int | TokenType::Uint | TokenType::Float) {
                t.origin_ident = Some(type_token.literal.clone());
                t.origin_type_kind = t.kind.clone();
            }

            t.end = self.prev().unwrap().end;
            return Ok(t);
        }

        // ptr<type>
        if self.consume(TokenType::Ptr) {
            self.must(TokenType::LeftAngle)?;
            let value_type = self.parser_type()?;
            self.must(TokenType::RightAngle)?;

            t.kind = TypeKind::Ptr(Box::new(value_type));
            t.end = self.prev().unwrap().end;
            return Ok(t);
        }

        // [type]
        if self.consume(TokenType::LeftSquare) {
            let element_type = self.parser_type()?;
            self.must(TokenType::RightSquare)?;

            t.kind = TypeKind::Vec(Box::new(element_type));
            t.end = self.prev().unwrap().end;
            return Ok(t);
        }

        // vec<type>
        if self.consume(TokenType::Vec) {
            self.must(TokenType::LeftAngle)?;
            let element_type = self.parser_type()?;
            self.must(TokenType::RightAngle)?;

            t.kind = TypeKind::Vec(Box::new(element_type));
            t.end = self.prev().unwrap().end;
            return Ok(t);
        }

        // map<type,type>
        if self.consume(TokenType::Map) {
            self.must(TokenType::LeftAngle)?;
            let key_type = self.parser_type()?;
            self.must(TokenType::Comma)?;
            let value_type = self.parser_type()?;
            self.must(TokenType::RightAngle)?;

            t.kind = TypeKind::Map(Box::new(key_type), Box::new(value_type));
            t.end = self.prev().unwrap().end;
            return Ok(t);
        }

        // set<type>
        if self.consume(TokenType::Set) {
            self.must(TokenType::LeftAngle)?;
            let element_type = self.parser_type()?;
            self.must(TokenType::RightAngle)?;

            t.kind = TypeKind::Set(Box::new(element_type));
            t.end = self.prev().unwrap().end;
            return Ok(t);
        }

        // tup<type, type, ...>
        if self.consume(TokenType::Tup) {
            self.must(TokenType::LeftAngle)?;
            let mut elements = Vec::new();

            loop {
                let element_type = self.parser_type()?;
                elements.push(element_type);

                if !self.consume(TokenType::Comma) {
                    break;
                }
            }
            self.must(TokenType::RightAngle)?;

            t.kind = TypeKind::Tuple(elements, 0);
            t.end = self.prev().unwrap().end;
            return Ok(t);
        }

        // chan<type>
        if self.consume(TokenType::Chan) {
            self.must(TokenType::LeftAngle)?;
            let element_type = self.parser_type()?;
            self.must(TokenType::RightAngle)?;

            t.kind = TypeKind::Chan(Box::new(element_type));
            t.end = self.prev().unwrap().end;
            return Ok(t);
        }

        // arr<type,length>
        if self.consume(TokenType::Arr) {
            self.must(TokenType::LeftAngle)?;
            let element_type = self.parser_type()?;
            self.must(TokenType::Comma)?;
            let length_token = self.must(TokenType::IntLiteral)?;

            let length = length_token
                .literal
                .parse::<u64>()
                .map_err(|_| SyntaxError(length_token.start, length_token.end, "array length must be a valid integer".to_string()))?;

            if length == 0 {
                return Err(SyntaxError(
                    length_token.start,
                    length_token.end,
                    "array length must be greater than 0".to_string(),
                ));
            }
            self.must(TokenType::RightAngle)?;

            t.kind = TypeKind::Arr(length, Box::new(element_type));
            t.end = self.prev().unwrap().end;
            return Ok(t);
        }

        // tuple (type, type)
        if self.consume(TokenType::LeftParen) {
            let mut elements = Vec::new();
            loop {
                let element_type = self.parser_type()?;
                elements.push(element_type);
                if !self.consume(TokenType::Comma) {
                    break;
                }
            }
            self.must(TokenType::RightParen)?;

            t.kind = TypeKind::Tuple(elements, 0);
            t.end = self.prev().unwrap().end;
            return Ok(t);
        }

        // {Type:Type} or {Type}
        if self.consume(TokenType::LeftCurly) {
            let key_type = self.parser_type()?;

            if self.consume(TokenType::Colon) {
                // map 类型
                let value_type = self.parser_type()?;
                self.must(TokenType::RightCurly)?;

                t.kind = TypeKind::Map(Box::new(key_type), Box::new(value_type));

                t.end = self.prev().unwrap().end;
                return Ok(t);
            } else {
                // set 类型
                self.must(TokenType::RightCurly)?;

                t.kind = TypeKind::Set(Box::new(key_type));

                t.end = self.prev().unwrap().end;
                return Ok(t);
            }
        }

        // struct { field_type field_name = default_value }
        if self.consume(TokenType::Struct) {
            self.must(TokenType::LeftCurly)?;

            let mut properties = Vec::new();

            while !self.is(TokenType::RightCurly) {
                let field_type = self.parser_type()?;
                let field_name = self.must(TokenType::Ident)?.literal.clone();

                let mut default_value = None;

                // 默认值支持
                if self.consume(TokenType::Equal) {
                    let expr = self.parser_expr()?;

                    // 不允许是函数定义
                    if let AstNode::FnDef(_) = expr.node {
                        return Err(SyntaxError(
                            expr.start,
                            expr.end,
                            "struct field default value cannot be a function definition".to_string(),
                        ));
                    }

                    default_value = Some(expr);
                }

                properties.push(TypeStructProperty {
                    type_: field_type.clone(),
                    key: field_name,
                    value: default_value,
                    start: field_type.start,
                    end: field_type.end,
                });

                self.must_stmt_end()?;
            }

            self.must(TokenType::RightCurly)?;

            t.kind = TypeKind::Struct("".to_string(), 0, properties);
            t.end = self.prev().unwrap().end;
            return Ok(t);
        }

        // fn(Type, Type, ...):T!
        if self.consume(TokenType::Fn) {
            self.must(TokenType::LeftParen)?;
            let mut param_types = Vec::new();

            if !self.consume(TokenType::RightParen) {
                loop {
                    let param_type = self.parser_type()?;
                    param_types.push(param_type);

                    if !self.consume(TokenType::Comma) {
                        break;
                    }
                }
                self.must(TokenType::RightParen)?;
            }

            let return_type = if self.consume(TokenType::Colon) {
                self.parser_type()?
            } else {
                Type::new(TypeKind::Void)
            };

            let errable = if self.consume(TokenType::Not) { true } else { false };

            t.kind = TypeKind::Fn(Box::new(TypeFn {
                name: "".to_string(),
                param_types,
                return_type,
                rest: false,
                tpl: false,
                errable,
            }));
            t.end = self.prev().unwrap().end;
            return Ok(t);
        }

        // ident foo = 12
        if self.is(TokenType::Ident) {
            self.set_current_token_type(SemanticTokenType::TYPE);
            let first = self.must(TokenType::Ident)?.clone();

            // handle param
            if !self.type_params_table.is_empty() && self.type_params_table.contains_key(&first.literal) {
                t.kind = TypeKind::Param(first.literal.clone());
                t.origin_ident = Some(first.literal.clone());
                t.origin_type_kind = t.kind.clone();
                return Ok(t);
            }

            // handle alias (package.ident)
            let mut second = None;
            if self.consume(TokenType::Dot) {
                self.set_current_token_type(SemanticTokenType::TYPE);
                second = Some(self.must(TokenType::Ident)?);
            }

            let ident = if let Some(second_token) = second {
                second_token.clone()
            } else {
                first.clone()
            };
            let mut alias = TypeAlias {
                ident: ident.literal,
                import_as: if second.is_some() { Some(first.literal.clone()) } else { None },
                symbol_id: None,
                args: None,
            };
            t.origin_ident = Some(alias.ident.clone());
            if let Some(import_as) = &alias.import_as {
                t.origin_ident = Some(format!("{}.{}", import_as, alias.ident.clone()));
            }

            // alias<arg1, arg2, ...>
            if self.consume(TokenType::LeftAngle) {
                let mut args = Vec::new();
                loop {
                    args.push(self.parser_single_type()?);
                    if !self.consume(TokenType::Comma) {
                        break;
                    }
                }
                self.must(TokenType::RightAngle)?;
                alias.args = Some(args);
            }

            t.kind = TypeKind::Alias(Box::new(alias));
            t.end = self.prev().unwrap().end;
            return Ok(t);
        }

        return Err(SyntaxError(self.peek().start, self.peek().end, "Type definition exception".to_string()));
    }

    fn parser_type(&mut self) -> Result<Type, SyntaxError> {
        let t = self.parser_single_type()?;

        // Type|Type or Type?
        if !self.is(TokenType::Or) && !self.is(TokenType::Question) {
            return Ok(t);
        }

        // T?, T? 后面不在允许直接携带 |
        if self.consume(TokenType::Question) {
            let mut union_t = Type::default();
            union_t.status = ReductionStatus::Undo;
            union_t.start = self.peek().start;
            let mut elements = Vec::new();
            elements.push(t);

            let null_type = Type::new(TypeKind::Null);
            elements.push(null_type);

            union_t.kind = TypeKind::Union(false, elements);
            union_t.end = self.prev().unwrap().end;
            return Ok(union_t);
        }

        if self.is(TokenType::Or) {
            return Err(SyntaxError(
                self.peek().start,
                self.peek().end,
                "union type only be declared in type alias".to_string(),
            ));
        }

        return Ok(t);
    }

    fn parser_type_alias_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let mut stmt = self.stmt_new();

        self.must(TokenType::Type)?;
        self.set_current_token_type(SemanticTokenType::TYPE);
        let ident_token = self.must(TokenType::Ident)?;
        let alias_ident = ident_token.clone();

        // T<arg1, arg2>
        let mut alias_params = Vec::new();
        if self.consume(TokenType::LeftAngle) {
            if self.is(TokenType::RightAngle) {
                return Err(SyntaxError(self.peek().start, self.peek().end, "type alias params cannot be empty".to_string()));
            }

            // 临时保存当前的 type_params_table
            self.type_params_table = HashMap::new();

            loop {
                let ident = self.must(TokenType::Ident)?.literal.clone();
                let mut param = GenericsParam::new(ident.clone());

                // 可选的泛型类型约束 <T:t1|t2, U:t1|t2>
                if self.consume(TokenType::Colon) {
                    param.constraints.0 = false;
                    loop {
                        let t = self.parser_single_type()?;

                        param.constraints.1.push(t);
                        if !self.consume(TokenType::Or) {
                            break;
                        }
                    }
                }

                alias_params.push(param);

                self.type_params_table.insert(ident.clone(), ident.clone());

                if !self.consume(TokenType::Comma) {
                    break;
                }
            }

            self.must(TokenType::RightAngle)?;
        }

        self.must(TokenType::Equal)?;

        let type_expr = if self.consume(TokenType::Struct) {
            self.must(TokenType::LeftCurly)?;

            let mut properties = Vec::new();
            while !self.is(TokenType::RightCurly) {
                let field_type = self.parser_type()?;
                let field_name = self.must(TokenType::Ident)?.literal.clone();

                let mut default_value = None;

                // 默认值支持
                if self.consume(TokenType::Equal) {
                    let expr = self.parser_expr()?;

                    // 不允许是函数定义
                    if let AstNode::FnDef(_) = expr.node {
                        return Err(SyntaxError(
                            expr.start,
                            expr.end,
                            "struct field default value cannot be a function definition".to_string(),
                        ));
                    }

                    default_value = Some(expr);
                }

                properties.push(TypeStructProperty {
                    type_: field_type.clone(),
                    key: field_name,
                    value: default_value,
                    start: field_type.start,
                    end: field_type.end,
                });

                self.must_stmt_end()?;
            }

            self.must(TokenType::RightCurly)?;

            Type::undo_new(TypeKind::Struct("".to_string(), 0, properties))
        } else {
            let mut alias_type = self.parser_single_type()?;

            if self.consume(TokenType::Question) {
                let mut elements = Vec::new();
                elements.push(alias_type);
                elements.push(Type::new(TypeKind::Null));

                // 不能再接 |
                if self.is(TokenType::Or) {
                    return Err(SyntaxError(
                        self.peek().start,
                        self.peek().end,
                        "union type declaration cannot use '?'".to_string(),
                    ));
                }

                alias_type = Type::undo_new(TypeKind::Union(false, elements));
            } else if self.consume(TokenType::Or) {
                let mut elements = Vec::new();
                elements.push(alias_type);

                loop {
                    let t = self.parser_single_type()?;
                    elements.push(t);

                    if !self.consume(TokenType::Or) {
                        break;
                    }
                }

                alias_type = Type::undo_new(TypeKind::Union(false, elements));
            }

            alias_type
        };

        // 恢复之前的 type_params_table
        self.type_params_table = HashMap::new();

        stmt.node = AstNode::TypeAlias(Arc::new(Mutex::new(TypeAliasStmt {
            ident: alias_ident.literal,
            symbol_start: alias_ident.start,
            symbol_end: alias_ident.end,
            params: alias_params,
            type_expr,
            symbol_id: None,
        })));
        stmt.end = self.prev().unwrap().end;

        Ok(stmt)
    }

    fn expr_to_type_alias(&self, left_expr: &Expr, generics_args: Option<Vec<Type>>) -> Type {
        let mut t = Type::default();
        t.status = ReductionStatus::Undo;
        t.start = self.peek().start;
        t.end = self.peek().end;

        // 根据左值表达式类型构造 TypeAlias
        let alias = match &left_expr.node {
            // 简单标识符: foo
            AstNode::Ident(ident, _) => {
                t.origin_ident = Some(ident.clone());
                t.origin_type_kind = TypeKind::Alias(Box::new(TypeAlias::default()));

                TypeAlias {
                    ident: ident.clone(),
                    import_as: None,
                    symbol_id: None,
                    args: generics_args,
                }
            }

            // 包选择器: pkg.foo
            AstNode::SelectExpr(left, key) => {
                if let AstNode::Ident(left_ident, _) = &left.node {
                    t.origin_ident = Some(key.clone());
                    t.origin_type_kind = TypeKind::Alias(Box::new(TypeAlias::default()));

                    TypeAlias {
                        ident: key.clone(),
                        import_as: Some(left_ident.clone()),
                        symbol_id: None,
                        args: generics_args,
                    }
                } else {
                    panic!("struct new left type exception");
                }
            }
            _ => panic!("struct new left type exception"),
        };

        t.kind = TypeKind::Alias(Box::new(alias));
        t
    }

    // 解析变量声明
    fn parser_var_decl(&mut self) -> Result<Arc<Mutex<VarDeclExpr>>, SyntaxError> {
        let var_type = self.parser_type()?;

        // 变量名必须是标识符
        let var_ident = self.must(TokenType::Ident)?;

        Ok(Arc::new(Mutex::new(VarDeclExpr {
            type_: var_type,
            ident: var_ident.literal.clone(),
            symbol_start: var_ident.start,
            symbol_end: var_ident.end,
            be_capture: false,
            heap_ident: None,
            symbol_id: None,
        })))
    }

    // 解析函数参数
    fn parser_params(&mut self, fn_decl: &mut AstFnDef) -> Result<(), SyntaxError> {
        self.must(TokenType::LeftParen)?;

        if self.consume(TokenType::RightParen) {
            return Ok(());
        }

        loop {
            if self.consume(TokenType::Ellipsis) {
                fn_decl.rest_param = true;
            }

            let param = self.parser_var_decl()?;
            fn_decl.params.push(param);

            // 可变参数必须是最后一个参数
            if fn_decl.rest_param && !self.is(TokenType::RightParen) {
                return Err(SyntaxError(
                    self.peek().start,
                    self.peek().end,
                    "can only use '...' as the final argument in the list".to_string(),
                ));
            }

            if !self.consume(TokenType::Comma) {
                break;
            }
        }

        self.must(TokenType::RightParen)?;
        Ok(())
    }

    // 解析二元表达式
    fn parser_binary(&mut self, left: Box<Expr>) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        expr.start = left.start;

        let operator_token = self.safe_advance()?.clone();

        // 获取运算符优先级
        let precedence = self.find_rule(operator_token.token_type.clone()).infix_precedence;

        let right = self.parser_precedence_expr(precedence.next().unwrap(), TokenType::Unknown)?;

        expr.node = AstNode::Binary(token_to_expr_op(&operator_token.token_type), left, right);
        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_left_angle_is_type_args(&mut self, left: &Box<Expr>) -> bool {
        // 保存当前解析位置, 为后面的错误恢复做准备
        let current_pos = self.current;

        // 必须是标识符或选择器表达式
        match &left.node {
            AstNode::Ident(..) => (),
            AstNode::SelectExpr(left, _) => {
                // 选择器的左侧必须是标识符
                if !matches!(left.node, AstNode::Ident(..)) {
                    return false;
                }
            }
            _ => return false,
        }

        // 跳过 <
        self.must(TokenType::LeftAngle).unwrap();

        // 尝试解析第一个类型
        if let Err(_) = self.parser_type() {
            // 类型解析存在错误
            self.current = current_pos;
            return false;
        }

        // 检查是否直接以 > 结束 (大多数情况)
        if self.is(TokenType::RightAngle) {
            self.current = current_pos;
            return true;
        }

        if self.consume(TokenType::Comma) {
            // 处理多个类型参数的情况
            loop {
                if let Err(_) = self.parser_type() {
                    self.current = current_pos;
                    return false;
                }

                if !self.consume(TokenType::Comma) {
                    break;
                }
            }

            if !self.is(TokenType::RightAngle) {
                self.current = current_pos;
                return false;
            }

            // type args 后面不能紧跟 { 或 (, 这两者通常是 generics params
            if !self.next_is(1, TokenType::LeftCurly) && !self.next_is(1, TokenType::LeftParen) {
                self.current = current_pos;
                return false;
            }

            self.current = current_pos;
            return true;
        }

        self.current = current_pos;
        return false;
    }

    fn parser_type_args_expr(&mut self, left: Box<Expr>) -> Result<Box<Expr>, SyntaxError> {
        assert!(self.is(TokenType::LeftAngle));

        let mut expr = self.expr_new();
        expr.start = left.start;

        // 解析泛型参数
        let mut generics_args = Vec::new();
        if self.consume(TokenType::LeftAngle) {
            loop {
                let t = self.parser_type()?;
                generics_args.push(t);

                if !self.consume(TokenType::Comma) {
                    break;
                }
            }
            self.must(TokenType::RightAngle)?;
        }

        // 判断下一个符号
        if self.is(TokenType::LeftParen) {
            // 函数调用
            let mut call = AstCall {
                return_type: Type::default(),
                left,
                generics_args,
                args: Vec::new(),
                spread: false,
            };

            call.args = self.parser_args(&mut call)?;

            expr.node = AstNode::Call(call);
            return Ok(expr);
        }

        // 结构体初始化
        assert!(self.is(TokenType::LeftCurly));
        let t = self.expr_to_type_alias(&left, Some(generics_args));

        self.parser_struct_new(t)
    }

    fn parser_struct_new(&mut self, type_: Type) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        let mut properties = Vec::new();

        self.must(TokenType::LeftCurly)?;

        if !self.consume(TokenType::RightCurly) {
            loop {
                let key = self.must(TokenType::Ident)?.literal.clone();

                self.must(TokenType::Equal)?;

                let value = self.parser_expr()?;

                properties.push(StructNewProperty {
                    type_: Type::default(), // 类型会在语义分析阶段填充
                    key,
                    start: value.start,
                    end: value.end,
                    value,
                });

                if !self.consume(TokenType::Comma) {
                    break;
                }
            }

            self.consume(TokenType::StmtEof);
            self.must(TokenType::RightCurly)?;
        }

        expr.node = AstNode::StructNew(String::new(), type_, properties);
        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_unary(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        let operator_token = self.safe_advance()?.clone();

        let operator = match operator_token.token_type {
            TokenType::Not => ExprOp::Not,
            TokenType::Minus => {
                // 检查是否可以直接合并成字面量
                if self.is(TokenType::IntLiteral) {
                    let int_token = self.must(TokenType::IntLiteral)?;
                    expr.node = AstNode::Literal(TypeKind::Int, format!("-{}", int_token.literal));
                    return Ok(expr);
                }

                if self.is(TokenType::FloatLiteral) {
                    let float_token = self.must(TokenType::FloatLiteral)?;
                    expr.node = AstNode::Literal(TypeKind::Float, format!("-{}", float_token.literal));
                    return Ok(expr);
                }

                ExprOp::Neg
            }
            TokenType::Tilde => ExprOp::Bnot,
            TokenType::And => ExprOp::La,
            TokenType::Star => ExprOp::Ia,
            _ => {
                return Err(SyntaxError(
                    operator_token.start,
                    operator_token.end,
                    format!("unknown unary operator '{}'", operator_token.literal),
                ));
            }
        };

        let operand = self.parser_precedence_expr(SyntaxPrecedence::Unary, TokenType::Unknown)?;
        expr.node = AstNode::Unary(operator, operand);
        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_catch_expr(&mut self, left: Box<Expr>) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        expr.start = left.start;
        self.must(TokenType::Catch)?;

        let error_ident = self.must(TokenType::Ident)?;

        let catch_err = VarDeclExpr {
            ident: error_ident.literal.clone(),
            symbol_start: error_ident.start,
            symbol_end: error_ident.end,
            type_: Type::default(), // 实际上就是 error type
            be_capture: false,
            heap_ident: None,
            symbol_id: None,
        };

        let catch_body = self.parser_body()?;

        expr.node = AstNode::Catch(left, Arc::new(Mutex::new(catch_err)), catch_body);
        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_as_expr(&mut self, left: Box<Expr>) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        self.must(TokenType::As)?;

        let target_type = self.parser_single_type()?;

        expr.start = left.start;
        expr.node = AstNode::As(target_type, left);
        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_match_is_expr(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        self.must(TokenType::Is)?;

        // 确保在 match 表达式中使用 is
        if !self.match_cond {
            return Err(SyntaxError(
                self.peek().start,
                self.peek().end,
                "is type must be specified in the match expression".to_string(),
            ));
        }

        let target_type = self.parser_single_type()?;

        expr.node = AstNode::MatchIs(target_type);
        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_is_expr(&mut self, left: Box<Expr>) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        self.must(TokenType::Is)?;

        let target_type = self.parser_single_type()?;

        expr.start = left.start;
        expr.node = AstNode::Is(target_type, left);
        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_left_paren_expr(&mut self) -> Result<Box<Expr>, SyntaxError> {
        self.must(TokenType::LeftParen)?;

        // 先尝试解析为普通表达式
        let expr = self.parser_expr()?;

        // 如果直接遇到右括号,说明是普通的括号表达式
        if self.consume(TokenType::RightParen) {
            return Ok(expr);
        }

        // 否则应该是元组表达式
        self.must(TokenType::Comma)?;

        let mut elements = Vec::new();
        elements.push(expr);

        // 继续解析剩余的元素
        loop {
            let element = self.parser_expr()?;
            elements.push(element);

            if !self.consume(TokenType::Comma) {
                break;
            }
        }

        self.must(TokenType::RightParen)?;

        let mut tuple_expr = self.expr_new();
        tuple_expr.node = AstNode::TupleNew(elements);
        tuple_expr.end = self.prev().unwrap().end;

        Ok(tuple_expr)
    }

    fn parser_literal(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        let literal_token = self.safe_advance()?;

        let kind = token_to_type_kind(&literal_token.token_type);

        expr.node = AstNode::Literal(kind, literal_token.literal.clone());
        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_is_tuple_typedecl(&self, current: usize) -> bool {
        let t = &self.token_db[self.token_indexes[current]];
        assert_eq!(t.token_type, TokenType::LeftParen, "tuple type decl start left param");

        // param is left paren, so close + 1 = 1,
        let mut close = 1;
        let mut pos = current;

        while t.token_type != TokenType::Eof {
            pos += 1;
            let t = &self.token_db[self.token_indexes[pos]];

            if t.token_type == TokenType::LeftParen {
                close += 1;
            }

            if t.token_type == TokenType::RightParen {
                close -= 1;
                if close == 0 {
                    break;
                }
            }
        }

        if close > 0 {
            return false;
        }

        // (...) ident; ) 的 下一符号如果是 ident 就表示 (...) 里面是 tuple typedecl
        let t = &self.token_db[self.token_indexes[pos + 1]];
        if t.token_type != TokenType::Ident {
            return false;
        }

        return true;
    }

    fn parser_ident_expr(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        let ident_token = self.must(TokenType::Ident)?;

        expr.node = AstNode::Ident(ident_token.literal.clone(), None);
        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_access(&mut self, left: Box<Expr>) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();

        self.must(TokenType::LeftSquare)?;
        let key = self.parser_expr()?;
        self.must(TokenType::RightSquare)?;

        expr.start = left.start;
        expr.node = AstNode::AccessExpr(left, key);
        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_select_expr(&mut self, left: Box<Expr>) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();

        self.must(TokenType::Dot)?;

        let property_token = self.must(TokenType::Ident)?;
        expr.start = left.start;
        expr.node = AstNode::SelectExpr(left, property_token.literal.clone());
        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_args(&mut self, call: &mut AstCall) -> Result<Vec<Box<Expr>>, SyntaxError> {
        self.must(TokenType::LeftParen)?;
        let mut args = Vec::new();

        // 无调用参数
        if self.consume(TokenType::RightParen) {
            return Ok(args);
        }

        loop {
            if self.is(TokenType::RightParen) {
                break;
            }

            if self.consume(TokenType::Ellipsis) {
                call.spread = true;
            }

            let expr = self.parser_expr()?;
            args.push(expr);

            // 可变参数必须是最后一个参数
            if call.spread && !self.is(TokenType::RightParen) {
                return Err(SyntaxError(
                    self.peek().start,
                    self.peek().end,
                    "can only use '...' as the final argument in the list".to_string(),
                ));
            }

            // call args 结尾可能存在 , 或者 ) 可以避免换行符识别异常，所以 parser 需要支持最后一个 TokenComma 可选情况
            if self.is(TokenType::RightParen) {
                break;
            } else {
                self.must(TokenType::Comma)?;
            }
        }

        self.must(TokenType::RightParen)?;
        Ok(args)
    }

    fn parser_call_expr(&mut self, left: Box<Expr>) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        expr.start = left.start;

        let mut call = AstCall {
            return_type: Type::default(),
            left,
            args: Vec::new(),
            generics_args: Vec::new(),
            spread: false,
        };

        call.args = self.parser_args(&mut call)?;

        expr.node = AstNode::Call(call);
        expr.end = self.prev().unwrap().end;
        Ok(expr)
    }

    fn parser_else_if(&mut self) -> Result<Vec<Box<Stmt>>, SyntaxError> {
        let mut stmt_list = Vec::new();
        stmt_list.push(self.parser_if_stmt()?);
        Ok(stmt_list)
    }

    fn parser_if_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let mut stmt = self.stmt_new();

        self.must(TokenType::If)?;

        let condition = self.parser_expr_with_precedence()?;
        let consequent = self.parser_body()?;

        let alternate = if self.consume(TokenType::Else) {
            if self.is(TokenType::If) {
                self.parser_else_if()?
            } else {
                self.parser_body()?
            }
        } else {
            Vec::new() // default empty vec
        };

        stmt.node = AstNode::If(condition, consequent, alternate);
        stmt.end = self.prev().unwrap().end;

        Ok(stmt)
    }

    fn is_for_tradition_stmt(&self) -> Result<bool, SyntaxError> {
        let mut semicolon_count = 0;
        let mut close = 0;
        let mut pos = self.current;
        let current_line = self.token_db[self.token_indexes[pos]].line;

        while pos < self.token_indexes.len() {
            let t = &self.token_db[self.token_indexes[pos]];

            if t.token_type == TokenType::Eof {
                return Err(SyntaxError(self.peek().start, self.peek().end, "unexpected end of file".to_string()));
            }

            if close == 0 && t.token_type == TokenType::StmtEof {
                semicolon_count += 1;
            }

            if t.token_type == TokenType::LeftCurly {
                close += 1;
            }

            if t.token_type == TokenType::RightCurly {
                close -= 1;
            }

            if t.line != current_line {
                break;
            }

            pos += 1;
        }

        if semicolon_count != 0 && semicolon_count != 2 {
            return Err(SyntaxError(
                self.peek().start,
                self.peek().end,
                "for statement must have two semicolons".to_string(),
            ));
        }

        Ok(semicolon_count == 2)
    }

    fn is_type_begin_stmt(&mut self) -> bool {
        // var/any/int/float/bool/string
        if self.is_basic_type() {
            return true;
        }

        if self.is(TokenType::Any) {
            return true;
        }

        // {int}/{int:int} 或 [int]
        if self.is(TokenType::LeftCurly) || self.is(TokenType::LeftSquare) {
            return true;
        }

        if self.is(TokenType::Ptr) {
            return true;
        }

        // 内置复合类型
        if matches!(
            self.peek().token_type,
            TokenType::Arr | TokenType::Map | TokenType::Tup | TokenType::Vec | TokenType::Set | TokenType::Chan
        ) {
            return true;
        }

        // fndef type (stmt 维度禁止了匿名 fndef, 所以这里一定是 fndef type)
        if self.is(TokenType::Fn) && self.next_is(1, TokenType::LeftParen) {
            return true;
        }

        // person a 连续两个 ident， 第一个 ident 一定是类型 ident
        if self.is(TokenType::Ident) && self.next_is(1, TokenType::Ident) {
            return true;
        }

        // package.ident foo = xxx
        if self.is(TokenType::Ident) && self.next_is(1, TokenType::Dot) && self.next_is(2, TokenType::Ident) && self.next_is(3, TokenType::Ident) {
            return true;
        }

        // person|i8 a
        if self.is(TokenType::Ident) && self.next_is(1, TokenType::Or) {
            return true;
        }

        // package.ident|i8 foo = xxx
        if self.is(TokenType::Ident) && self.next_is(1, TokenType::Dot) && self.next_is(2, TokenType::Ident) && self.next_is(3, TokenType::Or) {
            return true;
        }

        // person<[i8]> foo
        if self.is(TokenType::Ident) && self.next_is(1, TokenType::LeftAngle) {
            return true;
        }

        // person.foo<[i8]>
        if self.is(TokenType::Ident) && self.next_is(1, TokenType::Dot) && self.next_is(2, TokenType::Ident) && self.next_is(3, TokenType::LeftAngle) {
            return true;
        }

        // (var_a, var_b) = (1, 2)
        // (custom, int, int, (int, int), map) a = xxx
        if self.is(TokenType::LeftParen) && self.parser_is_tuple_typedecl(self.current) {
            return true;
        }

        false
    }

    fn parser_for_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        self.must(TokenType::For)?;

        let mut stmt = self.stmt_new();

        // 通过找 ; 号的形式判断, 必须要有两个 ; 才会是 tradition
        // for int i = 1; i <= 10; i+=1
        if self.is_for_tradition_stmt()? {
            let init = self.parser_for_init_stmt()?;
            self.must(TokenType::StmtEof)?;

            let cond = self.parser_expr_with_precedence()?;
            self.must(TokenType::StmtEof)?;

            let update = self.parser_for_init_stmt()?;

            let body = self.parser_body()?;

            stmt.node = AstNode::ForTradition(init, cond, update, body);
            stmt.end = self.prev().unwrap().end;

            return Ok(stmt);
        }

        // for k,v in map {}
        if self.is(TokenType::Ident) && (self.next_is(1, TokenType::Comma) || self.next_is(1, TokenType::In)) {
            let first_ident = self.must(TokenType::Ident)?;
            let first = VarDeclExpr {
                type_: Type::default(),
                ident: first_ident.literal.clone(),
                symbol_start: first_ident.start,
                symbol_end: first_ident.end,
                be_capture: false,
                symbol_id: None,
                heap_ident: None,
            };

            let second = if self.consume(TokenType::Comma) {
                let second_ident = self.must(TokenType::Ident)?;
                Some(Arc::new(Mutex::new(VarDeclExpr {
                    type_: Type::default(),
                    ident: second_ident.literal.clone(),
                    symbol_start: second_ident.start,
                    symbol_end: second_ident.end,
                    be_capture: false,
                    heap_ident: None,
                    symbol_id: None,
                })))
            } else {
                None
            };

            self.must(TokenType::In)?;
            let iterate = self.parser_precedence_expr(SyntaxPrecedence::TypeCast, TokenType::Unknown)?;
            let body = self.parser_body()?;

            stmt.node = AstNode::ForIterator(iterate, Arc::new(Mutex::new(first)), second, body);

            return Ok(stmt);
        }

        // for (condition) {}
        let condition = self.parser_expr_with_precedence()?;
        let body = self.parser_body()?;

        stmt.node = AstNode::ForCond(condition, body);

        Ok(stmt)
    }

    fn parser_assign(&mut self, left: Box<Expr>) -> Result<Box<Stmt>, SyntaxError> {
        let mut stmt = self.stmt_new();

        let stmt_start = left.start.clone();

        // 简单赋值
        if self.consume(TokenType::Equal) {
            let right = self.parser_expr()?;

            stmt.node = AstNode::Assign(left, right);
            stmt.start = stmt_start;
            stmt.end = self.prev().unwrap().end;

            return Ok(stmt);
        }

        // 复合赋值
        let t = self.safe_advance()?.clone();
        if !t.is_complex_assign() {
            return Err(SyntaxError(t.start, t.end, format!("expected '=' actual '{}'", t.token_type)));
        }

        let mut right = self.expr_new();
        right.node = AstNode::Binary(token_to_expr_op(&t.token_type), left.clone(), self.parser_expr_with_precedence()?);
        right.end = self.prev().unwrap().end;

        stmt.node = AstNode::Assign(left, right);
        stmt.start = stmt_start;
        stmt.end = self.prev().unwrap().end;

        Ok(stmt)
    }

    fn parser_expr_begin_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let left = self.parser_expr()?;

        // 处理函数调用语句
        if let AstNode::Call(call) = left.node {
            if self.is(TokenType::Equal) {
                return Err(SyntaxError(self.peek().start, self.peek().end, "call expr cannot assign".to_string()));
            }

            let mut stmt = self.stmt_new();
            stmt.node = AstNode::Call(call);
            stmt.start = left.start;
            stmt.end = self.prev().unwrap().end;
            return Ok(stmt);
        }

        // 处理 catch 语句
        if let AstNode::Catch(try_expr, catch_err, catch_body) = left.node {
            if self.is(TokenType::Equal) || self.is(TokenType::Catch) {
                return Err(SyntaxError(
                    self.peek().start,
                    self.peek().end,
                    "catch expr cannot assign or immediately next catch".to_string(),
                ));
            }

            let mut stmt = self.stmt_new();
            stmt.node = AstNode::Catch(try_expr, catch_err, catch_body);
            stmt.start = left.start;
            stmt.end = self.prev().unwrap().end;
            return Ok(stmt);
        }

        // 检查表达式完整性
        if self.is_stmt_eof() {
            return Err(SyntaxError(self.peek().start, self.peek().end, "expr incompleteness".to_string()));
        }

        // 处理赋值语句
        self.parser_assign(left)
    }

    fn parser_break_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let mut stmt = self.stmt_new();
        self.must(TokenType::Break)?;

        let expr = if !self.is_stmt_eof() && !self.is(TokenType::RightCurly) {
            Some(self.parser_expr()?)
        } else {
            None
        };

        stmt.node = AstNode::Break(expr);
        stmt.end = self.prev().unwrap().end;
        Ok(stmt)
    }

    fn parser_continue_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let mut stmt = self.stmt_new();
        self.must(TokenType::Continue)?;

        stmt.node = AstNode::Continue;
        stmt.end = self.prev().unwrap().end;
        Ok(stmt)
    }

    fn parser_return_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let mut stmt = self.stmt_new();
        self.must(TokenType::Return)?;

        let expr = if !self.is_stmt_eof() && !self.is(TokenType::RightCurly) {
            Some(self.parser_expr()?)
        } else {
            None
        };

        stmt.node = AstNode::Return(expr);
        stmt.end = self.prev().unwrap().end;
        Ok(stmt)
    }

    fn parser_import_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let mut stmt = self.stmt_new();
        self.must(TokenType::Import)?;

        let token = self.safe_advance()?.clone();
        let mut import_end = token.end;

        let (file, ast_package) = if token.token_type == TokenType::StringLiteral {
            (Some(token.literal.clone()), None)
        } else if token.token_type == TokenType::Ident {
            let mut package = vec![token.literal.clone()];
            while self.consume(TokenType::Dot) {
                let ident = self.must(TokenType::Ident)?;
                package.push(ident.literal.clone());
                import_end = ident.end;
            }
            (None, Some(package))
        } else {
            return Err(SyntaxError(token.start, token.end, "import token must be string or ident".to_string()));
        };

        let as_name = if self.consume(TokenType::As) {
            let t = self.safe_advance()?.clone();

            if !matches!(t.token_type, TokenType::Ident | TokenType::ImportStar) {
                return Err(SyntaxError(t.start, t.end, "import as token must be ident or *".to_string()));
            }
            t.literal.clone()
        } else {
            "".to_string()
        };

        stmt.node = AstNode::Import(ImportStmt {
            file,
            ast_package,
            as_name,
            module_type: 0,
            full_path: String::new(),
            package_conf: None,
            package_dir: String::new(),
            use_links: false,
            module_ident: String::new(),
            start: stmt.start,
            end: import_end,
        });

        stmt.end = self.prev().unwrap().end;
        Ok(stmt)
    }

    fn parser_vec_new(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        self.must(TokenType::LeftSquare)?;

        let mut elements = Vec::new();
        if !self.consume(TokenType::RightSquare) {
            loop {
                let element = self.parser_expr()?;
                elements.push(element);

                if !self.consume(TokenType::Comma) {
                    break;
                }
            }
            self.must(TokenType::RightSquare)?;
        }

        expr.node = AstNode::VecNew(elements, None, None);
        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_left_curly_expr(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();

        // parse empty curly
        self.must(TokenType::LeftCurly)?;
        if self.consume(TokenType::RightCurly) {
            expr.node = AstNode::EmptyCurlyNew;
            expr.end = self.prev().unwrap().end;
            return Ok(expr);
        }

        // parse first expr
        let key_expr = self.parser_expr()?;

        // if colon, parse map
        if self.consume(TokenType::Colon) {
            let mut elements = Vec::new();
            let value = self.parser_expr()?;

            elements.push(MapElement { key: key_expr, value });

            while self.consume(TokenType::Comma) {
                let key = self.parser_expr()?;
                self.must(TokenType::Colon)?;
                let value = self.parser_expr()?;
                elements.push(MapElement { key, value });
            }

            // skip stmt eof
            self.consume(TokenType::StmtEof);
            self.must(TokenType::RightCurly)?;

            expr.node = AstNode::MapNew(elements);
            expr.end = self.prev().unwrap().end;
            return Ok(expr);
        }

        // else is set
        let mut elements = Vec::new();
        elements.push(key_expr);

        while self.consume(TokenType::Comma) {
            let element = self.parser_expr()?;
            elements.push(element);
        }

        self.must(TokenType::RightCurly)?;
        expr.node = AstNode::SetNew(elements);
        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_fndef_expr(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        let start = self.peek().start;
        let end = self.peek().end;

        self.must(TokenType::Fn)?;

        let mut fndef = AstFnDef::default();
        fndef.start = start;
        fndef.end = end;

        if self.is(TokenType::Ident) {
            // fn expr 不能包含名称
            return Err(SyntaxError(self.peek().start, self.peek().end, "local fn must be anonymous".to_string()));
        } else {
            // gen unique lambda name
            let name = format!("{}{}", LOCAL_FN_NAME, self.lambda_index);
            self.lambda_index += 1;

            fndef.symbol_name = name.clone();
            fndef.fn_name = name;
        }

        self.parser_params(&mut fndef)?;

        // parse return type
        if self.consume(TokenType::Colon) {
            fndef.return_type = self.parser_type()?;
        } else {
            fndef.return_type = Type::new(TypeKind::Void);
        }

        if self.consume(TokenType::Not) {
            fndef.is_errable = true;
        }

        fndef.body = self.parser_body()?;
        expr.node = AstNode::FnDef(Arc::new(Mutex::new(fndef)));

        // parse immediately call fn expr
        if self.is(TokenType::LeftParen) {
            let mut call = AstCall {
                return_type: Type::default(),
                left: expr,
                args: Vec::new(),
                generics_args: Vec::new(),
                spread: false,
            };
            call.args = self.parser_args(&mut call)?;

            let mut call_expr = self.expr_new();
            call_expr.node = AstNode::Call(call);
            call_expr.end = self.prev().unwrap().end;
            return Ok(call_expr);
        }

        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_new_expr(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        self.must(TokenType::New)?;

        let t = self.parser_type()?;
        self.must(TokenType::LeftParen)?;
        let mut properties: Vec<StructNewProperty> = Vec::new();

        if !self.consume(TokenType::RightParen) {
            loop {
                if self.is(TokenType::RightParen) {
                    break;
                }

                let start = self.peek().start;
                let key = self.must(TokenType::Ident)?.clone();
                self.must(TokenType::Equal)?;
                let value = self.parser_expr()?;
                let end = value.end;

                properties.push(StructNewProperty {
                    type_: Type::default(),
                    key: key.literal,
                    value,
                    start,
                    end,
                });

                if self.is(TokenType::RightParen) {
                    break;
                } else {
                    self.must(TokenType::Comma)?;
                }
            }
            self.must(TokenType::RightParen)?;
        }

        expr.node = AstNode::New(t, properties);
        expr.end = self.prev().unwrap().end;

        Ok(expr)
    }

    fn parser_tuple_destr(&mut self) -> Result<Vec<Box<Expr>>, SyntaxError> {
        self.must(TokenType::LeftParen)?;

        let mut elements = Vec::new();
        loop {
            let element = if self.is(TokenType::LeftParen) {
                let mut expr = self.expr_new();
                expr.node = AstNode::TupleDestr(self.parser_tuple_destr()?);
                expr.end = self.prev().unwrap().end;
                expr
            } else {
                let mut expr = self.parser_expr()?;

                // 检查表达式是否可赋值
                if !expr.node.can_assign() {
                    return Err(SyntaxError(
                        self.peek().start,
                        self.peek().end,
                        "tuple destr src operand assign failed".to_string(),
                    ));
                }
                expr.end = self.prev().unwrap().end;
                expr
            };

            elements.push(element);

            if !self.consume(TokenType::Comma) {
                break;
            }
        }

        self.must(TokenType::RightParen)?;

        Ok(elements)
    }

    fn parser_var_tuple_destr(&mut self) -> Result<Vec<Box<Expr>>, SyntaxError> {
        self.must(TokenType::LeftParen)?;

        let mut elements = Vec::new();
        loop {
            let element = if self.is(TokenType::LeftParen) {
                let mut expr = self.expr_new();
                expr.node = AstNode::TupleDestr(self.parser_var_tuple_destr()?);
                expr.end = self.prev().unwrap().end;
                expr
            } else {
                let ident_token = self.must(TokenType::Ident)?.clone();
                let mut expr = self.expr_new();

                expr.node = AstNode::VarDecl(Arc::new(Mutex::new(VarDeclExpr {
                    type_: Type::default(),
                    ident: ident_token.literal.clone(),
                    symbol_start: ident_token.start,
                    symbol_end: ident_token.end,
                    be_capture: false,
                    heap_ident: None,
                    symbol_id: None,
                })));
                expr.end = self.prev().unwrap().end;
                expr
            };

            elements.push(element);

            if !self.consume(TokenType::Comma) {
                break;
            }
        }

        self.must(TokenType::RightParen)?;

        Ok(elements)
    }

    fn parser_var_begin_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let mut stmt = self.stmt_new();
        let type_decl = self.parser_type()?;

        // 处理 var (a, b) 形式
        if self.is(TokenType::LeftParen) {
            let tuple_destr = self.parser_var_tuple_destr()?;
            self.must(TokenType::Equal)?;
            let right = self.parser_expr()?;

            stmt.node = AstNode::VarTupleDestr(tuple_destr, right);
            stmt.end = self.prev().unwrap().end;
            return Ok(stmt);
        }

        // 处理 var a = 1 形式
        let ident = self.must(TokenType::Ident)?.clone();
        self.must(TokenType::Equal)?;

        stmt.node = AstNode::VarDef(
            Arc::new(Mutex::new(VarDeclExpr {
                type_: type_decl,
                ident: ident.literal,
                symbol_start: ident.start,
                symbol_end: ident.end,
                be_capture: false,
                heap_ident: None,
                symbol_id: None,
            })),
            self.parser_expr()?,
        );
        stmt.end = self.prev().unwrap().end;
        Ok(stmt)
    }

    fn parser_type_begin_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let mut stmt = self.stmt_new();
        let type_decl = self.parser_type()?;
        let ident = self.must(TokenType::Ident)?.clone();

        // 仅 var 支持元组解构
        if self.is(TokenType::LeftParen) {
            return Err(SyntaxError(
                self.peek().start,
                self.peek().end,
                "type begin stmt not support tuple destr".to_string(),
            ));
        }

        // 声明必须赋值
        self.must(TokenType::Equal)?;

        stmt.node = AstNode::VarDef(
            Arc::new(Mutex::new(VarDeclExpr {
                type_: type_decl,
                ident: ident.literal,
                symbol_start: ident.start,
                symbol_end: ident.end,
                be_capture: false,
                heap_ident: None,
                symbol_id: None,
            })),
            self.parser_expr()?,
        );
        stmt.end = self.prev().unwrap().end;
        Ok(stmt)
    }

    fn is_impl_fn(&self) -> bool {
        if self.is_basic_type() {
            return true;
        }

        if self.is(TokenType::Vec) || self.is(TokenType::Map) || self.is(TokenType::Set) {
            return true;
        }

        if self.is(TokenType::Chan) {
            return true;
        }

        if self.is(TokenType::Ident) && self.next_is(1, TokenType::Dot) {
            return true;
        }

        if self.is(TokenType::Ident) && self.next_is(1, TokenType::LeftParen) {
            return false;
        }

        if self.is(TokenType::Ident) && self.next_is(1, TokenType::LeftAngle) {
            let mut close = 1;
            let mut pos = self.current + 2;
            let current_line = self.token_db[self.token_indexes[self.current]].line;

            while pos < self.token_indexes.len() {
                let t = &self.token_db[self.token_indexes[pos]];

                if t.token_type == TokenType::Eof || t.token_type == TokenType::StmtEof || t.line != current_line {
                    return false;
                }

                if t.token_type == TokenType::LeftAngle {
                    close += 1;
                }

                if t.token_type == TokenType::RightAngle {
                    close -= 1;
                    if close == 0 {
                        break;
                    }
                }

                pos += 1;
            }

            if close > 0 {
                return false;
            }

            // 检查下一个 token
            if pos + 1 >= self.token_indexes.len() {
                return false;
            }

            let next = &self.token_db[self.token_indexes[pos + 1]];
            if next.token_type == TokenType::Dot {
                return true;
            }

            if next.token_type == TokenType::LeftParen {
                return false;
            }
        }

        false
    }

    fn is_impl_type(&mut self, kind: &TypeKind) -> bool {
        matches!(
            kind,
            TypeKind::String
                | TypeKind::Bool
                | TypeKind::Int
                | TypeKind::Uint
                | TypeKind::Int8
                | TypeKind::Int16
                | TypeKind::Int32
                | TypeKind::Int64
                | TypeKind::Uint8
                | TypeKind::Uint16
                | TypeKind::Uint32
                | TypeKind::Uint64
                | TypeKind::Float
                | TypeKind::Float32
                | TypeKind::Float64
                | TypeKind::Chan(..)
                | TypeKind::Vec(..)
                | TypeKind::Map(..)
                | TypeKind::Set(..)
                | TypeKind::Tuple(..)
                | TypeKind::Alias(..)
        )
    }

    fn parser_fndef_stmt(&mut self, mut fndef: AstFnDef) -> Result<Box<Stmt>, SyntaxError> {
        let mut stmt = self.stmt_new();
        fndef.start = self.peek().start;
        self.must(TokenType::Fn)?;

        // 检查是否是类型实现函数
        let is_impl_type = if self.is_impl_fn() {
            let temp_current = self.current; // 回退位置

            let first_token = self.safe_advance()?.clone();

            // 处理泛型参数
            if self.consume(TokenType::LeftAngle) {
                self.type_params_table = HashMap::new();
                fndef.generics_params = Some(Vec::new());

                loop {
                    let ident = self.must(TokenType::Ident)?.clone();

                    let mut param = GenericsParam::new(ident.literal.clone());

                    // 处理泛型约束 <T:t1|t2, U:t1|t2>
                    if self.consume(TokenType::Colon) {
                        param.constraints.0 = false;
                        loop {
                            let t = self.parser_single_type()?;
                            param.constraints.1.push(t);
                            if !self.consume(TokenType::Or) {
                                break;
                            }
                        }
                    }

                    if let Some(params) = &mut fndef.generics_params {
                        params.push(param);
                    }

                    self.type_params_table.insert(ident.literal.clone(), ident.literal.clone());

                    if !self.consume(TokenType::Comma) {
                        break;
                    }
                }

                self.must(TokenType::RightAngle)?;
            }

            self.current = temp_current;

            // 解析实现类型
            let impl_type = if first_token.token_type == TokenType::Ident {
                let mut t = Type::default();
                t.kind = TypeKind::Alias(Box::new(TypeAlias {
                    import_as: None,
                    ident: first_token.literal.clone(),
                    symbol_id: None,
                    args: None,
                }));
                t.impl_ident = Some(self.must(TokenType::Ident)?.literal.clone());

                if fndef.generics_params.is_some() {
                    self.must(TokenType::LeftAngle)?;
                    let mut args = Vec::new();

                    loop {
                        let param_type = self.parser_single_type()?;
                        assert!(matches!(param_type.kind, TypeKind::Param(_)));

                        if self.consume(TokenType::Colon) {
                            loop {
                                self.parser_single_type()?;
                                if !self.consume(TokenType::Or) {
                                    break;
                                }
                            }
                        }
                        args.push(param_type);

                        if !self.consume(TokenType::Comma) {
                            break;
                        }
                    }

                    self.must(TokenType::RightAngle)?;

                    if let TypeKind::Alias(alias) = &mut t.kind {
                        alias.args = Some(args);
                    }
                }
                t
            } else {
                let mut t = self.parser_single_type()?;
                t.impl_ident = Some(first_token.literal.clone());
                t
            };

            // 类型检查
            if !self.is_impl_type(&impl_type.kind) {
                return Err(SyntaxError(
                    self.peek().start,
                    self.peek().end,
                    format!("type '{}' cannot impl fn", impl_type.kind),
                ));
            }

            fndef.impl_type = impl_type;
            self.must(TokenType::Dot)?;

            true
        } else {
            false
        };

        // 处理函数名
        self.set_current_token_type(SemanticTokenType::FUNCTION);
        let ident = self.must(TokenType::Ident)?;

        fndef.symbol_name = ident.literal.clone();
        fndef.fn_name = ident.literal.clone();

        // 处理非实现类型的泛型参数
        if !is_impl_type && self.consume(TokenType::LeftAngle) {
            self.type_params_table = HashMap::new();
            fndef.generics_params = Some(Vec::new());

            loop {
                let ident = self.must(TokenType::Ident)?.literal.clone();
                let mut param = GenericsParam::new(ident.clone());

                if self.consume(TokenType::Colon) {
                    param.constraints.0 = false;
                    loop {
                        let t = self.parser_single_type()?;
                        param.constraints.1.push(t);
                        if !self.consume(TokenType::Or) {
                            break;
                        }
                    }
                }

                if let Some(params) = &mut fndef.generics_params {
                    params.push(param);
                }

                self.type_params_table.insert(ident.clone(), ident.clone());

                if !self.consume(TokenType::Comma) {
                    break;
                }
            }

            self.must(TokenType::RightAngle)?;
        }

        self.parser_params(&mut fndef)?;

        // 处理返回类型
        if self.consume(TokenType::Colon) {
            fndef.return_type = self.parser_type()?;
        } else {
            fndef.return_type = Type::new(TypeKind::Void);
            fndef.return_type.start = self.peek().start;
            fndef.return_type.end = self.peek().end;
        }

        if self.consume(TokenType::Not) {
            fndef.is_errable = true;
        }

        // tpl fn not body;
        if self.is_stmt_eof() {
            fndef.is_tpl = true;
            stmt.node = AstNode::FnDef(Arc::new(Mutex::new(fndef)));
            return Ok(stmt);
        }

        fndef.body = self.parser_body()?;

        fndef.end = if let Some(prev) = self.prev() { prev.end } else { self.peek().end };

        self.type_params_table = HashMap::new();

        stmt.node = AstNode::FnDef(Arc::new(Mutex::new(fndef)));
        stmt.end = self.prev().unwrap().end;
        Ok(stmt)
    }

    fn parser_label(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let mut fndef = AstFnDef::default();

        while self.is(TokenType::Label) {
            let token = self.must(TokenType::Label)?;

            if token.literal == "linkid" {
                if self.is(TokenType::Ident) {
                    let linkto = self.must(TokenType::Ident)?;
                    fndef.linkid = Some(linkto.literal.clone());
                } else {
                    let literal = self.must(TokenType::StringLiteral)?;
                    fndef.linkid = Some(literal.literal.clone());
                }
            } else if token.literal == "local" {
                fndef.is_private = true;
            } else if token.literal == "runtime_use" {
                self.must(TokenType::Ident)?;
            } else {
                // TODO 不认识的 label 进行 advance 直到下一个 label 开始
                return Err(SyntaxError(token.start, token.end, format!("unknown fn label '{}'", token.literal)));
            }
        }

        self.must(TokenType::StmtEof)?;

        if self.is(TokenType::Type) {
            self.parser_type_alias_stmt()
        } else if self.is(TokenType::Fn) {
            self.parser_fndef_stmt(fndef)
        } else {
            Err(SyntaxError(
                self.peek().start,
                self.peek().end,
                format!("the label can only be used in type alias or fn"),
            ))
        }
    }

    fn parser_let_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let mut stmt = self.stmt_new();
        self.must(TokenType::Let)?;

        let expr = self.parser_expr()?;

        // 确保是 as 表达式
        if !matches!(expr.node, AstNode::As(..)) {
            return Err(SyntaxError(expr.start, expr.end, "must be 'as' expr".to_string()));
        }

        stmt.node = AstNode::Let(expr);
        stmt.end = self.prev().unwrap().end;
        Ok(stmt)
    }

    fn parser_throw_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let mut stmt = self.stmt_new();
        self.must(TokenType::Throw)?;

        stmt.node = AstNode::Throw(self.parser_expr()?);
        stmt.end = self.prev().unwrap().end;
        Ok(stmt)
    }

    fn parser_left_paren_begin_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        // 保存当前位置以便回退
        let current_pos = self.current;

        // 尝试解析元组解构
        self.must(TokenType::LeftParen)?;
        let _ = self.parser_expr()?;
        let is_comma = self.is(TokenType::Comma);

        // 回退到开始位置
        self.current = current_pos;

        if is_comma {
            // 元组解构赋值语句
            let mut stmt = self.stmt_new();
            let mut left = self.expr_new();
            left.node = AstNode::TupleDestr(self.parser_tuple_destr()?);
            left.end = self.prev().unwrap().end;

            self.must(TokenType::Equal)?;
            let right = self.parser_expr()?;

            stmt.node = AstNode::Assign(left, right);
            stmt.end = self.prev().unwrap().end;
            Ok(stmt)
        } else {
            // 普通表达式语句
            self.parser_expr_begin_stmt()
        }
    }

    fn parser_global_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let stmt = if self.is(TokenType::Var) {
            self.parser_var_begin_stmt()?
        } else if self.is_type_begin_stmt() {
            self.parser_type_begin_stmt()?
        } else if self.is(TokenType::Label) {
            self.parser_label()?
        } else if self.is(TokenType::Fn) {
            self.parser_fndef_stmt(AstFnDef::default())?
        } else if self.is(TokenType::Import) {
            self.parser_import_stmt()?
        } else if self.is(TokenType::Type) {
            self.parser_type_alias_stmt()?
        } else {
            return Err(SyntaxError(
                self.peek().start,
                self.peek().end,
                format!("global statement cannot start with '{}'", self.peek().literal),
            ));
        };

        self.must_stmt_end()?;

        Ok(stmt)
    }

    /**
     * for (init_stmt; condition_expr; post_stmt)  
     * init 和 condition 是 stmt 结构，当前函数解析 init 和 condition 支持的所有类型的 stmt
     */
    fn parser_for_init_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let stmt = if self.is(TokenType::Var) {
            // var 声明语句
            self.parser_var_begin_stmt()?
        } else if self.is_type_begin_stmt() {
            // 类型声明语句
            self.parser_type_begin_stmt()?
        } else if self.is(TokenType::LeftParen) {
            // 元组解构赋值语句
            self.parser_left_paren_begin_stmt()?
        } else if self.is(TokenType::Ident) {
            // 普通赋值语句
            self.parser_expr_begin_stmt()?
        } else {
            return Err(SyntaxError(
                self.peek().start,
                self.peek().end,
                format!("for init statement cannot start with '{}'", self.peek().literal),
            ));
        };

        Ok(stmt)
    }

    fn parser_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let stmt = if self.is(TokenType::Var) {
            self.parser_var_begin_stmt()?
        } else if self.is_type_begin_stmt() {
            self.parser_type_begin_stmt()?
        } else if self.is(TokenType::LeftParen) {
            self.parser_left_paren_begin_stmt()?
        } else if self.is(TokenType::Throw) {
            self.parser_throw_stmt()?
        } else if self.is(TokenType::Let) {
            self.parser_let_stmt()?
        } else if self.is(TokenType::Ident) {
            self.parser_expr_begin_stmt()?
        } else if self.is(TokenType::Label) {
            self.parser_label()?
        } else if self.is(TokenType::If) {
            self.parser_if_stmt()?
        } else if self.is(TokenType::For) {
            self.parser_for_stmt()?
        } else if self.is(TokenType::Return) {
            self.parser_return_stmt()?
        } else if self.is(TokenType::Import) {
            self.parser_import_stmt()?
        } else if self.is(TokenType::Type) {
            self.parser_type_alias_stmt()?
        } else if self.is(TokenType::Continue) {
            self.parser_continue_stmt()?
        } else if self.is(TokenType::Break) {
            self.parser_break_stmt()?
        } else if self.is(TokenType::Go) {
            let expr = self.parser_go_expr()?;
            self.fake_new(expr)
        } else if self.is(TokenType::Match) {
            let expr = self.parser_match_expr()?;
            self.fake_new(expr)
        } else if self.is(TokenType::Select) {
            self.parser_select_stmt()?
        } else if self.is(TokenType::MacroIdent) {
            let expr = self.parser_expr_with_precedence()?;
            self.fake_new(expr)
        } else {
            return Err(SyntaxError(
                self.peek().start,
                self.peek().end,
                format!("local statement cannot start with '{}'", self.peek().literal),
            ));
        };

        self.must_stmt_end()?;

        Ok(stmt)
    }

    fn parser_precedence_expr(&mut self, precedence: SyntaxPrecedence, exclude: TokenType) -> Result<Box<Expr>, SyntaxError> {
        // 读取表达式前缀
        let rule = self.find_rule(self.peek().token_type.clone());

        let prefix_fn = rule
            .prefix
            .ok_or_else(|| SyntaxError(self.peek().start, self.peek().end, format!("<expr> expected, found '{}'", self.peek().literal)))?;

        let mut expr = prefix_fn(self)?;

        // 前缀表达式已经处理完成，判断是否有中缀表达式
        let mut token_type = self.parser_infix_token(&expr);
        if exclude != TokenType::Eof && token_type == exclude {
            return Ok(expr);
        }

        let mut infix_rule = self.find_rule(token_type);

        while infix_rule.infix_precedence >= precedence {
            let infix_fn = if let Some(infix) = infix_rule.infix {
                infix
            } else {
                panic!("invalid infix expression");
            };

            expr = infix_fn(self, expr)?;

            token_type = self.parser_infix_token(&expr);
            if exclude != TokenType::Eof && token_type == exclude {
                return Ok(expr);
            }

            infix_rule = self.find_rule(token_type);
        }

        Ok(expr)
    }

    fn is_struct_param_new_prefix(&self, current: usize) -> bool {
        let t = &self.token_db[self.token_indexes[current]];
        if t.token_type != TokenType::LeftAngle {
            return false;
        }

        let mut close = 1;
        let mut pos = current;

        while pos < self.token_indexes.len() {
            pos += 1;
            let t = &self.token_db[self.token_indexes[pos]];

            if t.token_type == TokenType::LeftAngle {
                close += 1;
            }

            if t.token_type == TokenType::RightAngle {
                close -= 1;
                if close == 0 {
                    break;
                }
            }

            if t.token_type == TokenType::Eof {
                return false;
            }
        }

        if close > 0 {
            return false;
        }

        // next is '{' ?
        if pos + 1 >= self.token_indexes.len() {
            return false;
        }

        self.token_db[self.token_indexes[pos + 1]].token_type == TokenType::LeftCurly
    }

    fn parser_is_struct_new_expr(&self) -> bool {
        // foo {}
        if self.is(TokenType::Ident) && self.next_is(1, TokenType::LeftCurly) {
            return true;
        }

        // foo.bar {}
        if self.is(TokenType::Ident) && self.next_is(1, TokenType::Dot) && self.next_is(2, TokenType::Ident) && self.next_is(3, TokenType::LeftCurly) {
            return true;
        }

        // foo<a, b> {}
        if self.is(TokenType::Ident) && self.next_is(1, TokenType::LeftAngle) {
            if self.is_struct_param_new_prefix(self.current + 1) {
                return true;
            }
        }

        // foo.bar<a, b> {}
        if self.is(TokenType::Ident) && self.next_is(1, TokenType::Dot) && self.next_is(2, TokenType::Ident) && self.next_is(3, TokenType::LeftAngle) {
            if self.is_struct_param_new_prefix(self.current + 3) {
                return true;
            }
        }

        false
    }

    fn parser_struct_new_expr(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let t = self.parser_type()?;
        self.parser_struct_new(t)
    }

    fn parser_expr_with_precedence(&mut self) -> Result<Box<Expr>, SyntaxError> {
        self.parser_precedence_expr(SyntaxPrecedence::Assign, TokenType::Unknown)
    }

    fn parser_macro_default_expr(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        self.must(TokenType::LeftParen)?;
        self.must(TokenType::RightParen)?;

        expr.node = AstNode::MacroDefault;
        expr.end = self.prev().unwrap().end;
        Ok(expr)
    }

    fn parser_macro_sizeof(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        self.must(TokenType::LeftParen)?;

        let target_type = self.parser_single_type()?;
        self.must(TokenType::RightParen)?;

        expr.node = AstNode::MacroSizeof(target_type);
        expr.end = self.prev().unwrap().end;
        Ok(expr)
    }

    fn parser_macro_reflect_hash(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        self.must(TokenType::LeftParen)?;

        let target_type = self.parser_single_type()?;
        self.must(TokenType::RightParen)?;

        expr.node = AstNode::MacroReflectHash(target_type);
        expr.end = self.prev().unwrap().end;
        Ok(expr)
    }

    fn coroutine_fn_closure(&mut self, call_expr: &Box<Expr>) -> AstFnDef {
        let mut fndef = AstFnDef::default();
        fndef.is_async = true;
        fndef.is_errable = true;
        fndef.params = Vec::new();
        fndef.return_type = Type::new(TypeKind::Void);

        let mut stmt_list = Vec::new();

        // var a = call(x, x, x)
        let mut vardef_stmt = self.stmt_new();
        vardef_stmt.node = AstNode::VarDef(
            Arc::new(Mutex::new(VarDeclExpr {
                type_: Type::default(),
                ident: "result".to_string(),
                symbol_start: 0,
                symbol_end: 0,
                be_capture: false,
                heap_ident: None,
                symbol_id: None,
            })),
            call_expr.clone(),
        );
        vardef_stmt.end = self.prev().unwrap().end;

        // co_return(&result)
        let mut call_stmt = self.stmt_new();
        let call = AstCall {
            return_type: Type::default(),
            left: Box::new(Expr::ident(fndef.start, fndef.end, "co_return".to_string(), None)),
            args: vec![Box::new(Expr {
                node: AstNode::Unary(ExprOp::La, Box::new(Expr::ident(fndef.start, fndef.end, "result".to_string(), None))),
                ..Default::default()
            })],
            generics_args: Vec::new(),
            spread: false,
        };
        call_stmt.node = AstNode::Call(call);
        call_stmt.end = self.prev().unwrap().end;

        stmt_list.push(vardef_stmt);
        stmt_list.push(call_stmt);
        fndef.body = stmt_list;

        fndef
    }

    fn coroutine_fn_void_closure(&mut self, call_expr: &Box<Expr>) -> AstFnDef {
        let mut fndef = AstFnDef::default();
        fndef.is_async = true;
        fndef.is_errable = true;
        fndef.params = Vec::new();
        fndef.return_type = Type::new(TypeKind::Void);

        let mut stmt_list = Vec::new();

        // call(x, x, x)
        let mut call_stmt = self.stmt_new();
        if let AstNode::Call(call) = &call_expr.node {
            call_stmt.node = AstNode::Call(call.clone());
            call_stmt.end = self.prev().unwrap().end;
        }
        stmt_list.push(call_stmt);
        fndef.body = stmt_list;
        fndef
    }

    fn parser_select_stmt(&mut self) -> Result<Box<Stmt>, SyntaxError> {
        let mut stmt = self.stmt_new();
        self.must(TokenType::Select)?;
        self.must(TokenType::LeftCurly)?;

        let mut cases = Vec::new();
        let mut has_default = false;
        let mut recv_count = 0;
        let mut send_count = 0;

        while !self.consume(TokenType::RightCurly) {
            let start = self.peek().start;
            let mut select_case = SelectCase {
                on_call: None,
                recv_var: None,
                handle_body: Vec::new(),
                is_recv: false,
                is_default: false,
                start,
                end: 0,
            };

            // 处理默认分支 _ -> { ... }
            if self.is(TokenType::Ident) && self.peek().literal == "_" {
                if has_default {
                    return Err(SyntaxError(
                        self.peek().start,
                        self.peek().end,
                        "select statement can only have one default case".to_string(),
                    ));
                }
                self.advance();
                self.must(TokenType::RightArrow)?;

                select_case.is_default = true;
                select_case.handle_body = self.parser_body()?;
                has_default = true;

                select_case.end = self.prev().unwrap().end;
                cases.push(select_case);

                self.must_stmt_end()?;
                continue;
            }

            // 解析通道操作调用
            let call_expr = self.parser_expr_with_precedence()?;

            if let AstNode::Call(call) = &call_expr.node {
                if let AstNode::SelectExpr(_left, key) = &call.left.node {
                    match key.as_str() {
                        "on_recv" => {
                            recv_count += 1;
                            select_case.is_recv = true;
                        }
                        "on_send" => {
                            send_count += 1;
                            select_case.is_recv = false;
                        }
                        _ => {
                            return Err(SyntaxError(
                                call_expr.start,
                                call_expr.end,
                                "only on_recv or on_send can be used in select case".to_string(),
                            ));
                        }
                    }
                    select_case.on_call = Some(call.clone());
                } else {
                    return Err(SyntaxError(call_expr.start, call_expr.end, "select case must be chan select call".to_string()));
                }
            } else {
                return Err(SyntaxError(call_expr.start, call_expr.end, "select case must be chan select call".to_string()));
            }

            self.must(TokenType::RightArrow)?;

            // 处理接收变量声明 -> msg { ... }
            if self.is(TokenType::Ident) {
                let ident_token = self.must(TokenType::Ident)?;
                select_case.recv_var = Some(Arc::new(Mutex::new(VarDeclExpr {
                    type_: Type::default(),
                    ident: ident_token.literal.clone(),
                    symbol_start: ident_token.start,
                    symbol_end: ident_token.end,
                    be_capture: false,
                    heap_ident: None,
                    symbol_id: None,
                })));
            }

            select_case.handle_body = self.parser_body()?;
            select_case.end = self.prev().unwrap().end;
            cases.push(select_case);

            self.must_stmt_end()?;
        }

        // 检查是否只有default分支
        if has_default && cases.len() == 1 {
            return Err(SyntaxError(self.peek().start, self.peek().end, "select must contains on_call case".to_string()));
        }

        //     Select(Vec<SelectCase>, bool, i16, i16), // (cases, has_default, send_count, recv_count)
        stmt.node = AstNode::Select(cases, has_default, send_count, recv_count);
        stmt.end = self.prev().unwrap().end;

        Ok(stmt)
    }

    fn parser_match_expr(&mut self) -> Result<Box<Expr>, SyntaxError> {
        self.must(TokenType::Match)?;
        let mut expr = self.expr_new();
        let mut subject = None;
        let mut cases = Vec::new();

        // match ({a, b, c}) {}
        if !self.is(TokenType::LeftCurly) {
            subject = Some(self.parser_expr_with_precedence()?);
            self.match_subject = true;
        }

        self.must(TokenType::LeftCurly)?;

        while !self.consume(TokenType::RightCurly) {
            self.match_cond = true;

            let start = self.peek().start;
            let mut cond_list = Vec::new();

            if subject.is_some() {
                loop {
                    let expr = self.parser_precedence_expr(SyntaxPrecedence::Assign, TokenType::Or)?;
                    cond_list.push(expr);
                    if !self.consume(TokenType::Or) {
                        break;
                    }
                }
            } else {
                cond_list.push(self.parser_expr()?);
            }

            self.must(TokenType::RightArrow)?;
            self.match_cond = false;

            let exec_body = if self.is(TokenType::LeftCurly) {
                self.parser_body()?
            } else {
                let exec_expr = self.parser_expr()?;

                // gen retrun stmt
                let mut stmt = self.stmt_new();
                stmt.node = AstNode::Break(Some(exec_expr.clone()));
                stmt.start = exec_expr.start.clone();
                stmt.end = exec_expr.end.clone();
                vec![stmt]
            };

            self.must_stmt_end()?;

            cases.push(MatchCase {
                cond_list,
                handle_body: exec_body,
                is_default: false,
                start,
                end: self.prev().unwrap().end,
            });
        }

        self.match_subject = false;
        expr.node = AstNode::Match(subject, cases);
        expr.end = self.prev().unwrap().end;
        Ok(expr)
    }

    fn parser_go_expr(&mut self) -> Result<Box<Expr>, SyntaxError> {
        self.must(TokenType::Go)?;
        let call_expr = self.parser_expr()?;

        // expr 的 type 必须是 call
        if !matches!(call_expr.node, AstNode::Call(_)) {
            return Err(SyntaxError(call_expr.start, call_expr.end, "go expr must be call".to_string()));
        }

        let mut expr = self.expr_new();
        expr.node = AstNode::MacroAsync(MacroAsyncExpr {
            origin_call: if let AstNode::Call(call) = &call_expr.node {
                Box::new(call.clone())
            } else {
                panic!("go expr must be call")
            },
            closure_fn: Arc::new(Mutex::new(self.coroutine_fn_closure(&call_expr))),
            closure_fn_void: Arc::new(Mutex::new(self.coroutine_fn_void_closure(&call_expr))),
            flag_expr: None,
            return_type: Type::new(TypeKind::Void),
        });

        Ok(expr)
    }

    fn parser_macro_async_expr(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        self.must(TokenType::LeftParen)?;

        let call_expr = self.parser_expr()?;
        let mut r#async = MacroAsyncExpr {
            origin_call: if let AstNode::Call(call) = &call_expr.node {
                Box::new(call.clone())
            } else {
                panic!("async expr must be call")
            },
            closure_fn: Arc::new(Mutex::new(self.coroutine_fn_closure(&call_expr))),
            closure_fn_void: Arc::new(Mutex::new(self.coroutine_fn_void_closure(&call_expr))),
            flag_expr: None,
            return_type: Type::new(TypeKind::Void),
        };

        if self.consume(TokenType::Comma) {
            r#async.flag_expr = Some(self.parser_expr()?);
        }
        self.must(TokenType::RightParen)?;

        expr.node = AstNode::MacroAsync(r#async);
        expr.end = self.prev().unwrap().end;
        Ok(expr)
    }

    fn parser_macro_ula_expr(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let mut expr = self.expr_new();
        self.must(TokenType::LeftParen)?;

        let src = self.parser_expr()?;
        self.must(TokenType::RightParen)?;

        expr.node = AstNode::MacroUla(src);
        expr.end = self.prev().unwrap().end;
        Ok(expr)
    }

    fn parser_macro_call(&mut self) -> Result<Box<Expr>, SyntaxError> {
        let token = self.must(TokenType::MacroIdent)?;

        // 根据宏名称选择对应的解析器
        match token.literal.as_str() {
            "sizeof" => self.parser_macro_sizeof(),
            "reflect_hash" => self.parser_macro_reflect_hash(),
            "default" => self.parser_macro_default_expr(),
            "async" => self.parser_macro_async_expr(),
            "ula" => self.parser_macro_ula_expr(),
            _ => Err(SyntaxError(token.start, token.end, format!("macro '{}' not defined", token.literal))),
        }
    }

    fn parser_expr(&mut self) -> Result<Box<Expr>, SyntaxError> {
        // 根据当前 token 类型选择对应的解析器
        if self.parser_is_struct_new_expr() {
            self.parser_struct_new_expr()
        } else if self.is(TokenType::Go) {
            self.parser_go_expr()
        } else if self.is(TokenType::Match) {
            self.parser_match_expr()
        } else if self.is(TokenType::Fn) {
            self.parser_fndef_expr()
        } else if self.is(TokenType::New) {
            self.parser_new_expr()
        } else {
            self.parser_expr_with_precedence()
        }
    }
}
