fn main() {}
// use std::path::Path;
// use std::fs::{read_dir, File};
// use std::io::Write;
// fn main() {
//     println!("cargo:rerun-if-changed={:?}", "src");
//     let mut file = File::create("src/shader.rs").expect("Create file");

//     let items: Vec<String> = read_dir("/home/maik/projects/rlsl/issues/.shaders/")
//         .expect("read")
//         .filter_map(|dir_entry| {
//             dir_entry
//                 .map(|dir_entry| {
//                     let file_name = format!(
//                         "{file_name:?}",
//                         file_name = Path::new(&dir_entry.file_name())
//                             .file_stem()
//                             .and_then(|s| s.to_str())
//                             .expect("stem")
//                     );
//                     let file_path = format!("{}", dir_entry.path().display());
//                     format!(
//                         "({name}, include_bytes!({path:?}).iter().cloned().collect())",
//                         name = file_name,
//                         path = file_path
//                     )
//                 })
//                 .ok()
//         })
//         .collect();
//         let code = format!(
//             "use std::collections::HashMap;\n
//             pub fn shader_map() -> HashMap<&'static str, Vec<u8>> {{\n
//                 [{items}].iter().cloned().collect()\n
//             }}", items = items.join(",\n")
//         );
// // let code = format!(
// //             "use std::collections::HashMap;\n
// //             pub fn shader_map() -> HashMap<&str, &str> {\n
// //                 [{items}].iter().cloned().collect()\n
// //             }", items = items.join(",\n")
// //         );
//         file.write_all(code.as_str().as_bytes());
// }
