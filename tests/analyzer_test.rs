use nls::analyzer::lexer::{Lexer, TokenType};
use nls::analyzer::syntax::*;
use ropey::Rope;

#[test]
fn test_rope() {
    let text = "你好\n世界"; // 8个字节(你=3字节,好=3字节,\n=1字节,世=3字节,界=3字节)
    let rope = Rope::from_str(text);

    assert_eq!(rope.try_byte_to_line(0).unwrap(), 0);  // 第一行
    assert_eq!(rope.try_byte_to_line(7).unwrap(), 1);  // 第二行
    assert!(rope.try_byte_to_line(14).is_err());       // 超出范围，应该返回错误
}

#[test]
fn test_analyzer() {
    let source = r#"
   import mod
    
    fn test() {
        int a = 1 + 12
        a = a / 12
        if a == 1 {
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] {
                println(i)
            }
        }
    
        var f = fn(int a, bool b) {}
    }
    "#
    .to_string();

    let (tokens, stmts, errors) = nls::analyzer::Analyzer::analyze(source);

    // 验证没有词法和语法错误
    assert!(errors.is_empty(), "Expected no analyzer errors, got: {:?}", errors);
}
