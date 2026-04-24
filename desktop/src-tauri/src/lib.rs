use serde::Serialize;

#[derive(Serialize)]
struct BackendInfo {
    host: String,
    port: u16,
}

#[tauri::command]
fn backend_info() -> BackendInfo {
    BackendInfo {
        host: "127.0.0.1".into(),
        port: 8765,
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![backend_info])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
