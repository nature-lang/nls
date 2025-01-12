use std::collections::HashMap;

use tower_lsp::lsp_types::SemanticTokenType;

use crate::nrs_lang::{Expr, Func, ImCompleteSemanticToken, Spanned};

pub const LEGEND_TYPE: &[SemanticTokenType] = &[
    SemanticTokenType::FUNCTION, // fn ident
    SemanticTokenType::VARIABLE, // variable ident
    SemanticTokenType::STRING, // string literal
    SemanticTokenType::COMMENT, // comment
    SemanticTokenType::NUMBER, // number literal
    SemanticTokenType::KEYWORD, //  所有的语法关键字，比如 var, if, then, else, fn ...
    SemanticTokenType::OPERATOR, // 运算符
    SemanticTokenType::PARAMETER, // function parameter ident
    SemanticTokenType::TYPE,          // 用于类型名称（如 int, float, string 等）
    SemanticTokenType::MACRO,         // 用于宏标识符
    SemanticTokenType::PROPERTY,      // struct property ident
    SemanticTokenType::NAMESPACE,     // package ident
];


pub fn semantic_token_type_index(token_type: SemanticTokenType) -> usize {
    if token_type == SemanticTokenType::FUNCTION { return 0; }
    if token_type == SemanticTokenType::VARIABLE { return 1; }
    if token_type == SemanticTokenType::STRING { return 2; }
    if token_type == SemanticTokenType::COMMENT { return 3; }
    if token_type == SemanticTokenType::NUMBER { return 4; }
    if token_type == SemanticTokenType::KEYWORD { return 5; }
    if token_type == SemanticTokenType::OPERATOR { return 6; }
    if token_type == SemanticTokenType::PARAMETER { return 7; }
    if token_type == SemanticTokenType::TYPE { return 8; }
    if token_type == SemanticTokenType::MACRO { return 9; }
    if token_type == SemanticTokenType::PROPERTY { return 10; }
    if token_type == SemanticTokenType::NAMESPACE { return 11; }

    panic!("unknown semantic token type: {:?}", token_type)
}

pub fn semantic_token_from_ast(ast: &HashMap<String, Func>) -> Vec<ImCompleteSemanticToken> {
    let mut semantic_tokens = vec![];

    ast.iter().for_each(|(_func_name, function)| {
        function.args.iter().for_each(|(_, span)| {
            semantic_tokens.push(ImCompleteSemanticToken {
                start: span.start,
                length: span.len(),
                token_type: LEGEND_TYPE
                    .iter()
                    .position(|item| item == &SemanticTokenType::PARAMETER)
                    .unwrap(),
            });
        });
        let (_, span) = &function.name;
        semantic_tokens.push(ImCompleteSemanticToken {
            start: span.start,
            length: span.len(),
            token_type: LEGEND_TYPE
                .iter()
                .position(|item| item == &SemanticTokenType::FUNCTION)
                .unwrap(),
        });
        semantic_token_from_expr(&function.body, &mut semantic_tokens);
    });

    semantic_tokens
}

pub fn semantic_token_from_expr(
    expr: &Spanned<Expr>,
    semantic_tokens: &mut Vec<ImCompleteSemanticToken>,
) {
    match &expr.0 {
        Expr::Error => {}
        Expr::Value(_) => {}
        Expr::List(_) => {}
        Expr::Local((_name, span)) => {
            semantic_tokens.push(ImCompleteSemanticToken {
                start: span.start,
                length: span.len(),
                token_type: LEGEND_TYPE
                    .iter()
                    .position(|item| item == &SemanticTokenType::VARIABLE)
                    .unwrap(),
            });
        }
        Expr::Let(_, rhs, rest, name_span) => {
            semantic_tokens.push(ImCompleteSemanticToken {
                start: name_span.start,
                length: name_span.len(),
                token_type: LEGEND_TYPE
                    .iter()
                    .position(|item| item == &SemanticTokenType::VARIABLE)
                    .unwrap(),
            });
            semantic_token_from_expr(rhs, semantic_tokens);
            semantic_token_from_expr(rest, semantic_tokens);
        }
        Expr::Then(first, rest) => {
            semantic_token_from_expr(first, semantic_tokens);
            semantic_token_from_expr(rest, semantic_tokens);
        }
        Expr::Binary(lhs, _op, rhs) => {
            semantic_token_from_expr(lhs, semantic_tokens);
            semantic_token_from_expr(rhs, semantic_tokens);
        }
        Expr::Call(expr, params) => {
            semantic_token_from_expr(expr, semantic_tokens);
            params.0.iter().for_each(|p| {
                semantic_token_from_expr(p, semantic_tokens);
            });
        }
        Expr::If(test, consequent, alternative) => {
            semantic_token_from_expr(test, semantic_tokens);
            semantic_token_from_expr(consequent, semantic_tokens);
            semantic_token_from_expr(alternative, semantic_tokens);
        }
        Expr::Print(expr) => {
            semantic_token_from_expr(expr, semantic_tokens);
        }
    }
}
