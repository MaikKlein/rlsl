use std::collections::HashMap;

pub fn shader_map() -> HashMap<&'static str, Vec<u8>> {
    [
        (
            "pass-through",
            include_bytes!("/home/maik/projects/rlsl/issues/.shaders/pass-through.spv")
                .iter()
                .cloned()
                .collect(),
        ),
        (
            "single-branch",
            include_bytes!("/home/maik/projects/rlsl/issues/.shaders/single-branch.spv")
                .iter()
                .cloned()
                .collect(),
        ),
        (
            "vertex",
            include_bytes!("/home/maik/projects/rlsl/issues/.shaders/vertex.spv")
                .iter()
                .cloned()
                .collect(),
        ),
    ].iter()
        .cloned()
        .collect()
}
