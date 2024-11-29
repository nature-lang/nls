use super::common::AnalyzerError;
use strum_macros::Display;

#[derive(Debug, Clone, PartialEq, Display)]
pub enum TokenType {
    #[strum(serialize = "unknown")]
    Unknown = 0,
    // 单字符标记
    #[strum(serialize = "(")]
    LeftParen,
    #[strum(serialize = ")")]
    RightParen,
    #[strum(serialize = "[")]
    LeftSquare,
    #[strum(serialize = "]")]
    RightSquare,
    #[strum(serialize = "{")]
    LeftCurly,
    #[strum(serialize = "}")]
    RightCurly,
    #[strum(serialize = "<")]
    LeftAngle,
    #[strum(serialize = "<")]
    LessThan,
    #[strum(serialize = ">")]
    RightAngle,

    #[strum(serialize = ",")]
    Comma,
    #[strum(serialize = ".")]
    Dot,
    #[strum(serialize = "-")]
    Minus,
    #[strum(serialize = "+")]
    Plus,
    #[strum(serialize = "...")]
    Ellipsis,
    #[strum(serialize = ":")]
    Colon,
    #[strum(serialize = ";")]
    Semicolon,
    #[strum(serialize = "/")]
    Slash,
    #[strum(serialize = "*")]
    Star,
    #[strum(serialize = "*")]
    ImportStar,
    #[strum(serialize = "%")]
    Percent,
    #[strum(serialize = "?")]
    Question,
    #[strum(serialize = "->")]
    RightArrow,

    // 一到两个字符的标记
    #[strum(serialize = "!")]
    Not,
    #[strum(serialize = "!=")]
    NotEqual,
    #[strum(serialize = "=")]
    Equal,
    #[strum(serialize = "==")]
    EqualEqual,
    #[strum(serialize = ">=")]
    GreaterEqual,
    #[strum(serialize = "<=")]
    LessEqual,
    #[strum(serialize = "&&")]
    AndAnd,
    #[strum(serialize = "||")]
    OrOr,

    // 复合赋值运算符
    #[strum(serialize = "+=")]
    PlusEqual,
    #[strum(serialize = "-=")]
    MinusEqual,
    #[strum(serialize = "*=")]
    StarEqual,
    #[strum(serialize = "/=")]
    SlashEqual,
    #[strum(serialize = "%=")]
    PercentEqual,
    #[strum(serialize = "&=")]
    AndEqual,
    #[strum(serialize = "|=")]
    OrEqual,
    #[strum(serialize = "^=")]
    XorEqual,
    #[strum(serialize = "<<=")]
    LeftShiftEqual,
    #[strum(serialize = ">>=")]
    RightShiftEqual,

    // 位运算
    #[strum(serialize = "~")]
    Tilde,
    #[strum(serialize = "&")]
    And,
    #[strum(serialize = "|")]
    Or,
    #[strum(serialize = "^")]
    Xor,
    #[strum(serialize = "<<")]
    LeftShift,
    #[strum(serialize = ">>")]
    RightShift,

    // 字面量
    #[strum(serialize = "ident_literal")]
    Ident,
    #[strum(serialize = "#")]
    Pound,
    #[strum(serialize = "macro_ident")]
    MacroIdent,
    #[strum(serialize = "fn_label")]
    FnLabel,
    #[strum(serialize = "string_literal")]
    StringLiteral,
    #[strum(serialize = "float_literal")]
    FloatLiteral,
    #[strum(serialize = "int_literal")]
    IntLiteral,

    // 类型
    #[strum(serialize = "string")]
    String,
    #[strum(serialize = "bool")]
    Bool,
    #[strum(serialize = "float")]
    Float,
    #[strum(serialize = "int")]
    Int,
    #[strum(serialize = "uint")]
    Uint,
    #[strum(serialize = "u8")]
    U8,
    #[strum(serialize = "u16")]
    U16,
    #[strum(serialize = "u32")]
    U32,
    #[strum(serialize = "u64")]
    U64,
    #[strum(serialize = "i8")]
    I8,
    #[strum(serialize = "i16")]
    I16,
    #[strum(serialize = "i32")]
    I32,
    #[strum(serialize = "i64")]
    I64,
    #[strum(serialize = "f32")]
    F32,
    #[strum(serialize = "f64")]
    F64,
    #[strum(serialize = "new")]
    New,

    // 内置复合类型
    #[strum(serialize = "arr")]
    Arr,
    #[strum(serialize = "vec")]
    Vec,
    #[strum(serialize = "map")]
    Map,
    #[strum(serialize = "tup")]
    Tup,
    #[strum(serialize = "set")]
    Set,
    #[strum(serialize = "chan")]
    Chan,

    // 关键字
    #[strum(serialize = "ptr")]
    Ptr,
    #[strum(serialize = "true")]
    True,
    #[strum(serialize = "false")]
    False,
    #[strum(serialize = "type")]
    Type,
    #[strum(serialize = "null")]
    Null,
    #[strum(serialize = "void")]
    Void,
    #[strum(serialize = "any")]
    Any,
    #[strum(serialize = "struct")]
    Struct,
    #[strum(serialize = "throw")]
    Throw,
    #[strum(serialize = "try")]
    Try,
    #[strum(serialize = "catch")]
    Catch,
    #[strum(serialize = "match")]
    Match,
    #[strum(serialize = "continue")]
    Continue,
    #[strum(serialize = "break")]
    Break,
    #[strum(serialize = "for")]
    For,
    #[strum(serialize = "in")]
    In,
    #[strum(serialize = "if")]
    If,
    #[strum(serialize = "else")]
    Else,
    #[strum(serialize = "else if")]
    ElseIf,
    #[strum(serialize = "var")]
    Var,
    #[strum(serialize = "let")]
    Let,
    #[strum(serialize = "is")]
    Is,
    #[strum(serialize = "sizeof")]
    Sizeof,
    #[strum(serialize = "reflect_hash")]
    ReflectHash,
    #[strum(serialize = "as")]
    As,
    #[strum(serialize = "fn")]
    Fn,
    #[strum(serialize = "import")]
    Import,
    #[strum(serialize = "return")]
    Return,
    #[strum(serialize = "go")]
    Go,
    #[strum(serialize = ";")]
    StmtEof,
    #[strum(serialize = "\0")]
    Eof,
}

#[derive(Debug, Clone)]
pub enum LeftAngleType {
    FnArgs,
    TypeArgs,
    LogicLt,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub literal: String,
    pub line: usize,
    pub start: usize, // start index
    pub end: usize,   // end index
    pub length: usize,
}

impl Token {
    pub fn new(token_type: TokenType, literal: String, start: usize, end: usize, line: usize) -> Self {
        let length = literal.len();
        Self {
            token_type,
            literal,
            line,
            start,
            end,
            length,
        }
    }

    pub fn is_complex_assign(&self) -> bool {
        matches!(
            self.token_type,
            TokenType::PercentEqual
                | TokenType::MinusEqual
                | TokenType::PlusEqual
                | TokenType::SlashEqual
                | TokenType::StarEqual
                | TokenType::OrEqual
                | TokenType::AndEqual
                | TokenType::XorEqual
                | TokenType::LeftShiftEqual
                | TokenType::RightShiftEqual
        )
    }

    pub fn debug(&self) -> String {
        format!(
            "Token {{ type: {:?}, literal: '{}', start: {}, end: {}, length: {} }}",
            self.token_type, self.literal, self.start, self.end, self.length
        )
    }
}

#[derive(Debug)]
pub struct Lexer {
    source: String,
    offset: usize,
    guard: usize,
    length: usize,
    space_prev: char,
    space_next: char,
    line: usize,
    errors: Vec<AnalyzerError>,
}

impl Lexer {
    pub fn new(source: String) -> Self {
        Lexer {
            offset: 0,
            guard: 0,
            source,
            length: 0,
            space_prev: '\0',
            space_next: '\0',
            line: 1,
            errors: Vec::new(),
        }
    }

    pub fn scan(&mut self) -> (Vec<Token>, Vec<AnalyzerError>) {
        let mut tokens = Vec::new();

        while !self.at_eof() {
            let has_newline = self.skip_space();
            if self.at_eof() {
                break;
            }

            if has_newline && !tokens.is_empty() {
                let prev_token = tokens.last().unwrap();

                // has newline 后如果上一个字符可以接受语句结束符，则必须插入语句结束符
                if self.need_stmt_end(prev_token) {
                    tokens.push(Token::new(
                        TokenType::StmtEof,
                        ";".to_string(),
                        prev_token.end,
                        prev_token.end + 1,
                        prev_token.line,
                    ));
                }
            }

            let next_token = self.item(&tokens);
            tokens.push(next_token);
        }

        tokens.push(Token::new(
            TokenType::Eof,
            "EOF".to_string(),
            self.offset,
            self.guard,
            self.line,
        ));

        (tokens, self.errors.clone())
    }

    fn ident_advance(&mut self) -> String {
        while !self.at_eof() && (self.is_alpha(self.peek_guard()) || self.is_number(self.peek_guard())) {
            self.guard_advance();
        }
        self.gen_word()
    }

    fn ident(&self, word: &str, _: usize) -> TokenType {
        match word {
            "as" => TokenType::As,
            "any" => TokenType::Any,
            "arr" => TokenType::Arr,
            "bool" => TokenType::Bool,
            "break" => TokenType::Break,
            "catch" => TokenType::Catch,
            "chan" => TokenType::Chan,
            "continue" => TokenType::Continue,
            "else" => TokenType::Else,
            "false" => TokenType::False,
            "float" => TokenType::Float,
            "fn" => TokenType::Fn,
            "for" => TokenType::For,
            "go" => TokenType::Go,
            "if" => TokenType::If,
            "import" => TokenType::Import,
            "in" => TokenType::In,
            "int" => TokenType::Int,
            "is" => TokenType::Is,
            "let" => TokenType::Let,
            "map" => TokenType::Map,
            "match" => TokenType::Match,
            "new" => TokenType::New,
            "null" => TokenType::Null,
            "ptr" => TokenType::Ptr,
            "return" => TokenType::Return,
            "set" => TokenType::Set,
            "string" => TokenType::String,
            "struct" => TokenType::Struct,
            "throw" => TokenType::Throw,
            "true" => TokenType::True,
            "try" => TokenType::Try,
            "tup" => TokenType::Tup,
            "type" => TokenType::Type,
            "uint" => TokenType::Uint,
            "var" => TokenType::Var,
            "vec" => TokenType::Vec,
            "void" => TokenType::Void,
            // 数字类型
            "f32" => TokenType::F32,
            "f64" => TokenType::F64,
            "i8" => TokenType::I8,
            "i16" => TokenType::I16,
            "i32" => TokenType::I32,
            "i64" => TokenType::I64,
            "u8" => TokenType::U8,
            "u16" => TokenType::U16,
            "u32" => TokenType::U32,
            "u64" => TokenType::U64,
            _ => TokenType::Ident,
        }
    }

    fn skip_space(&mut self) -> bool {
        let mut has_newline = false;

        if self.guard != self.offset {
            self.space_prev = self.peek_guard_prev();
        }

        while !self.at_eof() {
            match self.peek_guard() {
                ' ' | '\r' | '\t' => {
                    self.guard_advance();
                }
                '\n' => {
                    self.guard_advance();
                    has_newline = true;
                }
                '/' => {
                    if let Some(next_char) = self.peek_next() {
                        if next_char == '/' {
                            while !self.at_eof() && self.peek_guard() != '\n' {
                                self.guard_advance();
                            }
                        } else if next_char == '*' {
                            while !self.multi_comment_end() {
                                if self.at_eof() {
                                    // 直到完美结尾都没有找到注释闭合符号, 不需要做任何错误恢复，已经到达了文件的末尾
                                    self.errors.push(AnalyzerError {
                                        start: self.offset,
                                        end: self.guard,
                                        message: String::from("Unterminated comment"),
                                    });
                                }

                                self.guard_advance();
                            }
                            self.guard_advance(); // *
                            self.guard_advance(); // /
                        } else {
                            self.space_next = self.peek_guard();
                            return has_newline;
                        }
                    }
                }
                _ => {
                    self.space_next = self.peek_guard();
                    return has_newline;
                }
            }
        }

        has_newline
    }

    fn gen_word(&self) -> String {
        self.source[self.offset..self.guard].to_string()
    }

    fn is_string(&self, s: char) -> bool {
        matches!(s, '"' | '`' | '\'')
    }

    fn is_float(&self, word: &str) -> bool {
        let dot_count = word.chars().filter(|&c| c == '.').count();
        if word.ends_with('.') || dot_count > 1 {
            false
        } else {
            dot_count == 1
        }
    }

    fn is_alpha(&self, c: char) -> bool {
        c.is_ascii_alphabetic() || c == '_'
    }

    fn is_number(&self, c: char) -> bool {
        c.is_ascii_digit()
    }

    fn is_hex_number(&self, c: char) -> bool {
        c.is_ascii_hexdigit()
    }

    fn is_oct_number(&self, c: char) -> bool {
        ('0'..='7').contains(&c)
    }

    fn is_bin_number(&self, c: char) -> bool {
        c == '0' || c == '1'
    }

    fn at_eof(&self) -> bool {
        self.guard >= self.source.len()
    }

    fn guard_advance(&mut self) -> char {
        let c = self.peek_guard();
        if c == '\n' {
            self.line += 1;
        }
        self.guard += 1;
        self.length += 1;

        c // c 实现了 copy， 所以这里不会发生所有权转移，顶多就是 clone
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.at_eof() {
            return false;
        }

        if self.peek_guard() != expected {
            return false;
        }

        self.guard_advance();
        true
    }

    fn item(&mut self, tokens: &[Token]) -> Token {
        // reset by guard
        self.offset = self.guard;
        self.length = 0;

        // 检查标识符
        if self.is_alpha(self.peek_guard()) {
            let word = self.ident_advance();
            let token = Token::new(self.ident(&word, self.length), word, self.offset, self.guard, self.line);
            return token;
        }

        // 检查宏标识符
        if self.match_char('@') {
            let word = self.ident_advance();
            let word = word[1..].to_string(); // 跳过 @ 字符
            return Token::new(TokenType::MacroIdent, word, self.offset, self.guard, self.line);
        }

        // 检查函数标签
        if self.match_char('#') {
            let word = self.ident_advance();
            let word = word[1..].to_string(); // 跳过 # 字符
            return Token::new(TokenType::FnLabel, word, self.offset, self.guard, self.line);
        }

        // 检查数字
        if self.is_number(self.peek_guard()) {
            let word: String;

            // 处理 0 开头的特殊数字格式
            if self.peek_guard() == '0' {
                if let Some(next_char) = self.peek_next() {
                    match next_char {
                        'x' => {
                            let num = self.hex_number_advance();
                            let decimal = self.number_convert(&num, 16);
                            word = decimal.to_string();
                        }
                        'o' => {
                            let num = self.oct_number_advance();
                            let decimal = self.number_convert(&num, 8);
                            word = decimal.to_string();
                        }
                        'b' => {
                            let num = self.bin_number_advance();
                            let decimal = self.number_convert(&num, 2);
                            word = decimal.to_string();
                        }
                        _ => {
                            word = self.number_advance();
                        }
                    }
                } else {
                    word = self.number_advance();
                }
            } else {
                word = self.number_advance();
            }

            // 判断数字类型
            let token_type = if self.is_float(&word) {
                TokenType::FloatLiteral
            } else {
                TokenType::IntLiteral
            };

            return Token::new(token_type, word, self.offset, self.guard, self.line);
        }

        // 检查字符串
        if self.is_string(self.peek_guard()) {
            let str = self.string_advance(self.peek_guard());
            return Token::new(TokenType::StringLiteral, str, self.offset, self.guard, self.line);
        }

        // 处理特殊字符
        let special_type = self.special_char();
        assert!(special_type != TokenType::Eof, "special characters are not recognized");

        // 检查 import xxx as * 的特殊情况
        if special_type == TokenType::Star && !tokens.is_empty() {
            if let Some(prev_token) = tokens.last() {
                if prev_token.token_type == TokenType::As {
                    return Token::new(
                        TokenType::ImportStar,
                        self.gen_word(),
                        self.offset,
                        self.guard,
                        self.line,
                    );
                }
            }
        }

        Token::new(special_type, self.gen_word(), self.offset, self.guard, self.line)
    }

    fn multi_comment_end(&self) -> bool {
        self.source[self.guard..].starts_with("*/")
    }

    fn special_char(&mut self) -> TokenType {
        let c = self.guard_advance();
        match c {
            '(' => TokenType::LeftParen,
            ')' => TokenType::RightParen,
            '[' => TokenType::LeftSquare,
            ']' => TokenType::RightSquare,
            '{' => TokenType::LeftCurly,
            '}' => TokenType::RightCurly,
            ':' => TokenType::Colon,
            ';' => TokenType::StmtEof,
            ',' => TokenType::Comma,
            '?' => TokenType::Question,
            '%' => {
                if self.match_char('=') {
                    TokenType::PercentEqual
                } else {
                    TokenType::Percent
                }
            }
            '-' => {
                if self.match_char('=') {
                    TokenType::MinusEqual
                } else if self.match_char('>') {
                    TokenType::RightArrow
                } else {
                    TokenType::Minus
                }
            }
            '+' => {
                if self.match_char('=') {
                    TokenType::PlusEqual
                } else {
                    TokenType::Plus
                }
            }
            '/' => {
                if self.match_char('=') {
                    TokenType::SlashEqual
                } else {
                    TokenType::Slash
                }
            }
            '*' => {
                if self.match_char('=') {
                    TokenType::StarEqual
                } else {
                    TokenType::Star
                }
            }
            '.' => {
                if self.match_char('.') {
                    if self.match_char('.') {
                        return TokenType::Ellipsis;
                    } else {
                        self.errors.push(AnalyzerError {
                            start: self.offset,
                            end: self.guard,
                            message: String::from("Expected '...'"),
                        });

                        return TokenType::Ellipsis;
                    }
                } else {
                    TokenType::Dot
                }
            }
            '!' => {
                if self.match_char('=') {
                    TokenType::NotEqual
                } else {
                    TokenType::Not
                }
            }
            '=' => {
                if self.match_char('=') {
                    TokenType::EqualEqual
                } else {
                    TokenType::Equal
                }
            }
            '<' => {
                if self.match_char('<') {
                    if self.match_char('=') {
                        TokenType::LeftShiftEqual
                    } else {
                        TokenType::LeftShift
                    }
                } else if self.match_char('=') {
                    TokenType::LessEqual
                } else {
                    TokenType::LeftAngle
                }
            }
            '>' => {
                if self.match_char('=') {
                    TokenType::GreaterEqual
                } else if self.match_char('>') {
                    if self.match_char('=') {
                        TokenType::RightShiftEqual
                    } else {
                        TokenType::RightShift
                    }
                } else {
                    TokenType::RightAngle
                }
            }
            '&' => {
                if self.match_char('&') {
                    TokenType::AndAnd
                } else {
                    TokenType::And
                }
            }
            '|' => {
                if self.match_char('|') {
                    TokenType::OrOr
                } else {
                    TokenType::Or
                }
            }
            '~' => TokenType::Tilde,
            '^' => {
                if self.match_char('=') {
                    TokenType::XorEqual
                } else {
                    TokenType::Xor
                }
            }
            _ => {
                self.errors.push(AnalyzerError {
                    start: self.offset,
                    end: self.guard,
                    message: String::from("Unexpected character"),
                });
                TokenType::Unknown
            }
        }
    }

    fn string_advance(&mut self, close_char: char) -> String {
        // 跳过开始的 ""，但不计入 token 长度
        self.guard += 1;
        let escape_char = '\\';
        let mut result = String::new();

        while !self.at_eof() && self.peek_guard() != close_char {
            let mut guard = self.peek_guard();

            if guard == '\n' {
                self.errors.push(AnalyzerError {
                    start: self.offset,
                    end: self.guard,
                    message: String::from("string not terminated"),
                });
                return result; // 返回已经解析的字符串
            }

            // 处理转义字符
            if guard == escape_char {
                self.guard += 1;
                guard = self.peek_guard();

                // 将转义字符转换为实际字符
                guard = match guard {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    'b' => '\x08',
                    'f' => '\x0C',
                    'a' => '\x07',
                    'v' => '\x0B',
                    '0' => '\0',
                    '\\' | '\'' | '"' => guard,
                    _ => {
                        self.errors.push(AnalyzerError {
                            start: self.guard,
                            end: self.guard + 1,
                            message: format!("unknown escape char '{}'", guard),
                        });
                        guard
                    }
                };
            }

            result.push(guard);
            self.guard_advance();
        }

        // 跳过结束引号，但不计入 token 长度
        self.guard += 1;

        result
    }

    fn number_convert(&mut self, word: &str, base: u32) -> i64 {
        i64::from_str_radix(word, base).unwrap_or_else(|_| {
            self.errors.push(AnalyzerError {
                start: self.offset,
                end: self.guard,
                message: format!("Invalid number `{}`", word),
            });
            return 0;
        })
    }

    fn need_stmt_end(&self, prev_token: &Token) -> bool {
        matches!(
            prev_token.token_type,
            TokenType::ImportStar
                | TokenType::IntLiteral
                | TokenType::StringLiteral
                | TokenType::FloatLiteral
                | TokenType::Ident
                | TokenType::Break
                | TokenType::Continue
                | TokenType::Return
                | TokenType::True
                | TokenType::False
                | TokenType::RightParen
                | TokenType::RightSquare
                | TokenType::RightCurly
                | TokenType::RightAngle
                | TokenType::Bool
                | TokenType::Float
                | TokenType::F32
                | TokenType::F64
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
                | TokenType::String
                | TokenType::Null
        )
    }

    // 添加缺失的数字处理函数
    fn hex_number_advance(&mut self) -> String {
        self.guard += 2; // 跳过 "0x"
        self.offset = self.guard;

        while !self.at_eof() && self.is_hex_number(self.peek_guard()) {
            self.guard_advance();
        }
        self.gen_word()
    }

    fn oct_number_advance(&mut self) -> String {
        self.guard += 2; // 跳过 "0o"
        self.offset = self.guard;

        while !self.at_eof() && self.is_oct_number(self.peek_guard()) {
            self.guard_advance();
        }
        self.gen_word()
    }

    fn bin_number_advance(&mut self) -> String {
        self.guard += 2; // 跳过 "0b"
        self.offset = self.guard;

        while !self.at_eof() && self.is_bin_number(self.peek_guard()) {
            self.guard_advance();
        }
        self.gen_word()
    }

    fn number_advance(&mut self) -> String {
        while !self.at_eof() && (self.is_number(self.peek_guard()) || self.peek_guard() == '.') {
            self.guard_advance();
        }
        self.gen_word()
    }

    pub fn get_context(&self) -> String {
        let context_start = self.offset.saturating_sub(20);
        let context_end = (self.offset + 20).min(self.source.len());
        format!("...{}...", &self.source[context_start..context_end])
    }

    fn peek_guard(&self) -> char {
        if self.guard >= self.source.len() {
            panic!(
                "Unexpected end of file: guard index {} exceeds source length {}",
                self.guard,
                self.source.len()
            );
        }

        self.source.as_bytes()[self.guard] as char
    }
    fn peek_guard_prev(&self) -> char {
        self.source.as_bytes()[self.guard - 1] as char
    }

    fn peek_next(&self) -> Option<char> {
        Some(self.source.as_bytes()[self.guard + 1] as char)
    }

    pub fn debug_tokens(tokens: &[Token]) {
        println!("=== Tokens Debug ===");
        for (i, token) in tokens.iter().enumerate() {
            println!("[{}] {}", i, token.debug());
        }
        println!("=== End Tokens ===");
    }
}
