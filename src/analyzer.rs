pub mod common;
pub mod lexer; // 声明子模块
pub mod semantic;
pub mod symbol;
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
        let (token_db, token_indexes, lexter_errors) = lexer::Lexer::new(source.clone()).scan();
        dbg!("lexer_scan completed, errors count: {}", lexter_errors.len(), token_db.clone());

        let mut errors = Vec::new();
        errors.extend(lexter_errors);

        // 2. 语法分析,  并且进行了 tokens 进行了类型矫正
        let (stmts, sem_token_db, syntax_errors) = syntax::Syntax::new(token_db, token_indexes).parser();
        dbg!("syntax_parser completed, errors count: {}", syntax_errors.len());
        errors.extend(syntax_errors);

        // 3.  TODO pre global import handle

        // 3. 语义分析
        let mut sema = semantic::Semantic::new("test_pkg".to_string(), stmts);

        let (analyze_stmts, sema_errors) = sema.analyze();
        errors.extend(sema_errors);

        // 4. 类型分析
        (sem_token_db, analyze_stmts, errors)
    }
}
