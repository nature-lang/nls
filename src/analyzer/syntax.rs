use super::common::{AnalyzerErrors, AstNode, Expr, ExprOp, Stmt, Type, TypeKind};
use super::lexer::Token;
use super::lexer::TokenType;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

pub struct SyntaxError(String);

impl fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SyntaxError: {}", self.0)
    }
}

impl fmt::Debug for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SyntaxError: {}", self.0)
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

#[derive(Debug, Clone, Copy)]
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

pub struct Syntax {
    tokens: Vec<Token>,
    current: usize, // token index

    errors: AnalyzerErrors,

    // parser 阶段辅助记录当前的 type_param, 当进入到 fn body 或者 struct def 时可以准确识别当前是 type param 还是 alias, 仅仅使用到 key
    // 默认是一个空 hashmap
    parser_type_params_table: HashMap<String, String>,

    // 部分表达式只有在 match cond 中可以使用，比如 is T, n if n xxx 等, parser_match_cond 为 true 时，表示当前处于 match cond 中
    parser_match_cond: bool,
}

impl Syntax {
    // static method new, Syntax::new(tokens)
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens: tokens,
            current: 0,
            parser_type_params_table: HashMap::new(),
            parser_match_cond: false,
            errors: AnalyzerErrors::new(),
        }
    }

    fn advance(&mut self) -> &Token {
        assert!(
            self.current + 1 < self.tokens.len(),
            "Syntax::advance: current index out of range"
        );

        let token = &self.tokens[self.current];

        self.current += 1;
        return token;
    }

    fn peek(&self) -> &Token {
        return &self.tokens[self.current];
    }

    fn prev(&self) -> Option<&Token> {
        if self.current == 0 {
            return None;
        }

        return Some(&self.tokens[self.current - 1]);
    }

    fn is(&self, token_type: TokenType) -> bool {
        return self.peek().type_ == token_type;
    }

    fn consume(&mut self, token_type: TokenType) -> bool {
        if self.is(token_type) {
            self.advance();
            return true;
        }
        return false;
    }

    fn must(&mut self, expect: TokenType) -> Result<&Token, SyntaxError> {
        let token = self.peek(); // 对 self 进行了不可变借用

        if token.type_ != expect {
            let message = format!(
                "expected '{}' found '{}'",
                expect.to_string(),
                token.literal
            );

            // 对 self.errors 进行了可变借用, 不过 token 的生命周期在 push 之前已经结束了，所以这里可以进行可变借用
            self.errors.push(token.start, token.end, message.clone());

            return Err(SyntaxError(message));
        }

        self.advance();
        return Ok(self.prev().unwrap());
    }

    // 对应 parser_next
    fn next(&self, step: usize) -> Option<&Token> {
        if self.current + step >= self.tokens.len() {
            return None;
        }
        Some(&self.tokens[self.current + step])
    }

    // 对应 parser_next_is
    fn next_is(&self, step: usize, expect: TokenType) -> bool {
        match self.next(step) {
            Some(token) => token.type_ == expect,
            None => false,
        }
    }

    // 对应 parser_is_stmt_eof
    fn is_stmt_eof(&self) -> bool {
        // 注意:这里假设TokenType中有StmtEof和Eof这两个枚举值
        // 如果没有的话需要在TokenType中添加
        self.is(TokenType::StmtEof) || self.is(TokenType::Eof)
    }

    fn stmt_new(&self) -> Box<Stmt> {
        Box::new(Stmt {
            start: self.peek().start,
            end: self.peek().end,
            value: AstNode::None,
        })
    }

    fn expr_new_box(&self) -> Box<Expr> {
        Box::new(self.expr_new())
    }

    fn expr_new(&self) -> Expr {
        Expr {
            start: self.peek().start,
            end: self.peek().end,
            type_: Type::default(),
            target_type: Type::default(),
            value: Box::new(AstNode::None),
        }
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

        // stmt eof 失败。报告错误，并返回 false 即可
        // 获取前一个 token 的位置用于错误报告
        if let Some(prev_token) = self.prev() {
            self.errors.push(
                prev_token.start,
                prev_token.end,
                "expected ';' or '}' at end of statement".to_string(),
            );
        }

        Err(SyntaxError("expected ';' or '}' at end of statement".to_string()))
    }
}
