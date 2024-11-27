pub mod lexer; // 声明子模块
pub mod common;
pub mod syntax;

pub struct Analyzer {
    source: String,
    tokens: Vec<lexer::Token>,
}

impl Analyzer {
    pub fn new(source: String) -> Self {
        Self {
            source,
            tokens: Vec::new(),
        }
    }

    // 执行完整的分析流程
    pub fn analyze(&mut self) -> Result<(), String> {
        // 1. 词法分析
        let mut lexer = lexer::Lexer::new(self.source.clone());
        let (_, _) = lexer.scan();
        
        // 2. 语法分析 (后续添加)
        // self.parse()?;
        
        // 3. 语义分析 (后续添加)
        // self.check_semantics()?;
        
        Ok(())
    }
    
    // 获取分析结果
    pub fn get_tokens(&self) -> &[lexer::Token] {
        &self.tokens
    }
}