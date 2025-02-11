use tower_lsp::lsp_types::Position;
use ropey::Rope;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub fn offset_to_position(offset: usize, rope: &Rope) -> Option<Position> {
    let line = rope.try_char_to_line(offset).ok()?;
    let first_char_of_line = rope.try_line_to_char(line).ok()?;
    let column = offset - first_char_of_line;
    Some(Position::new(line as u32, column as u32))
}

pub fn position_to_offset(position: Position, rope: &Rope) -> Option<usize> {
    let line_char_offset = rope.try_line_to_char(position.line as usize).ok()?;
    let slice = rope.slice(0..line_char_offset + position.character as usize);
    Some(slice.len_bytes())
}

pub fn format_global_ident(prefix: String, ident: String)->String {
    // 如果 prefix 为空，则直接返回 ident
    if prefix.is_empty() {
        return ident;
    }
    format!("{prefix}.{ident}")
}

pub fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    t.hash(&mut hasher);
    hasher.finish()
}

pub fn align_up(n: u64, align: u64) -> u64 {
    if align == 0 {
        return n;
    }
    (n + align - 1) & !(align - 1)
}