import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";

const projectRoot = resolve(import.meta.dirname, "..");
const executable = resolve(
  process.env.CARGO_HOME ?? resolve(homedir(), ".cargo"),
  "bin",
  process.platform === "win32" ? "cargo-about.exe" : "cargo-about",
);

if (!existsSync(executable)) {
  console.error("cargo-aboutが見つかりません。`cargo install cargo-about --locked --features cli`を実行してください。");
  process.exit(1);
}

const result = spawnSync(executable, [
  "generate",
  "--locked",
  "--target", "x86_64-pc-windows-msvc",
  "--manifest-path", "src-tauri/Cargo.toml",
  "src-tauri/about.hbs",
  "--output-file", "licenses/CARGO_DEPENDENCIES.html",
], {
  cwd: projectRoot,
  stdio: "inherit",
  windowsHide: true,
});

if (result.error) throw result.error;
process.exit(result.status ?? 1);
