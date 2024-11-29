use nls::analyzer::lexer::{Lexer, TokenType};
use nls::analyzer::syntax::*;
use nls::analyzer::common::*;

#[test]
fn test_lexer() {
    let source = r#"
        int i = 0
        for 20 > i {
            println(i)
            i = i + 1
        }
        print('for end, i=', i)
    "#
    .to_string();

    let mut l = Lexer::new(source);
    let (tokens, _) = l.scan();

    // 移除过滤器代码
    Lexer::debug_tokens(&tokens);

    let expected_types = vec![
        TokenType::Int,
        TokenType::Ident, // i
        TokenType::Equal,
        TokenType::IntLiteral, // 0
        TokenType::StmtEof,    // 语句结束

        TokenType::For,
        TokenType::IntLiteral, // 20
        TokenType::RightAngle, // >
        TokenType::Ident,      // i
        TokenType::LeftCurly,
        TokenType::Ident, // println
        TokenType::LeftParen,
        TokenType::Ident, // i
        TokenType::RightParen,
        TokenType::StmtEof, // 语句结束
        TokenType::Ident,   // i
        TokenType::Equal,
        TokenType::Ident, // i
        TokenType::Plus,
        TokenType::IntLiteral, // 1
        TokenType::StmtEof,    // 语句结束
        TokenType::RightCurly,
        TokenType::StmtEof,    // 语句结束
        TokenType::Ident, // print
        TokenType::LeftParen,
        TokenType::StringLiteral, // 'for end, i='
        TokenType::Comma,
        TokenType::Ident, // i
        TokenType::RightParen,
        TokenType::Eof,     // 文件结束
    ];

    assert_eq!(tokens.len(), expected_types.len());

    let mut i = 0;
    for (token, expected_type) in tokens.iter().zip(expected_types.iter()) {
        assert_eq!(token.token_type, *expected_type, "token type mismatch at index {}", i);
        i += 1;
    }
}

#[test]
fn test_syntax() {
    let source = r#"

    if b == 24 {{

    "#
    .to_string();

    let mut lexer = Lexer::new(source);
    let (tokens, lexer_errors) = lexer.scan();
    assert!(lexer_errors.is_empty(), "Expected no lexer errors");

    let mut syntax = Syntax::new(tokens);
    let (_stmts, syntax_errors) = syntax.parser();
    assert!(syntax_errors.is_empty(), "Expected no syntax errors");
}
