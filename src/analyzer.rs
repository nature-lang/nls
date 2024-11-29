pub mod lexer; // 声明子模块
pub mod common;
pub mod syntax;

use common::{AnalyzerError, Stmt};

pub struct Analyzer {
    // source: String,
    // tokens: Vec<lexer::Token>,
    // errors: Vec<AnalyzerError>,
    // stmts: Vec<Box<Stmt>>,
}

impl Analyzer {
    // 执行完整的分析流程
    pub fn analyze(source: String) -> (Vec<lexer::Token>, Vec<Box<Stmt>>, Vec<AnalyzerError>) {
        // 1. 词法分析
        let mut lexer = lexer::Lexer::new(source.clone());
        let (tokens, lexter_errors) = lexer.scan();
        dbg!("lexer_scan completed, errors count: {}", lexter_errors.len());

        let mut errors = Vec::new();
        errors.extend(lexter_errors);

        // 2. 语法分析 
        let mut syntax = syntax::Syntax::new(tokens.clone());
        let (stmts, syntax_errors) = syntax.parser();
        dbg!("syntax_parser completed, errors count: {}", syntax_errors.len());

        errors.extend(syntax_errors);

        // 3. 语义分析
        // self.check_semantics()?;

        // 4. 类型分析

        (tokens, stmts, errors)
    }
}