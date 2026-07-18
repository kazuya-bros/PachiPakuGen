import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { delimiter, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectDirectory = dirname(scriptDirectory);
const tauriCli = join(projectDirectory, "node_modules", "@tauri-apps", "cli", "tauri.js");

const sourcePathKey = Object.keys(process.env).find(key => key.toLowerCase() === "path") ?? "PATH";
const targetPathKey = process.platform === "win32" ? "Path" : "PATH";
const currentPath = process.env[sourcePathKey] ?? "";
const cargoExecutable = process.platform === "win32" ? "cargo.exe" : "cargo";
const rustupExecutable = join(
  homedir(),
  ".cargo",
  "bin",
  process.platform === "win32" ? "rustup.exe" : "rustup",
);

function rustupWhich(tool) {
  if (!existsSync(rustupExecutable)) return null;
  const result = spawnSync(rustupExecutable, ["which", tool], {
    encoding: "utf8",
    windowsHide: true,
  });
  const resolved = result.status === 0 ? result.stdout.trim() : "";
  return resolved && existsSync(resolved) ? resolved : null;
}

// ~/.cargo/bin の cargo.exe はrustupへのシンボリックリンクである場合がある。
// Tauriのnative CLIからリンクを解決できない環境向けに、toolchain内の実体を最優先する。
const toolchainCargo = rustupWhich("cargo");
const toolchainRustc = rustupWhich("rustc");
const cargoHomes = [process.env.CARGO_HOME, join(homedir(), ".cargo")].filter(Boolean);
const cargoBins = [...new Set([
  ...(toolchainCargo ? [dirname(toolchainCargo)] : []),
  ...cargoHomes.map(home => join(home, "bin")),
])]
  .filter(bin => existsSync(join(bin, cargoExecutable)));

if (cargoBins.length === 0) {
  console.error(`RustのCargoが見つかりません: ${join(homedir(), ".cargo", "bin", cargoExecutable)}`);
  console.error("Rustupをインストールしてから、もう一度実行してください。");
  process.exit(1);
}

const pathEntries = currentPath.split(delimiter).filter(Boolean);
const normalizedEntries = new Set(pathEntries.map(entry => entry.toLowerCase()));
const missingCargoBins = cargoBins.filter(bin => !normalizedEntries.has(bin.toLowerCase()));
const env = { ...process.env };
for (const key of Object.keys(env)) {
  if (key.toLowerCase() === "path") delete env[key];
}
env[targetPathKey] = [...missingCargoBins, ...pathEntries].join(delimiter);
if (toolchainCargo) env.CARGO = toolchainCargo;
if (toolchainRustc && existsSync(toolchainRustc)) env.RUSTC = toolchainRustc;

const metadata = spawnSync(cargoExecutable, ["metadata", "--no-deps", "--format-version", "1"], {
  cwd: join(projectDirectory, "src-tauri"),
  env,
  encoding: "utf8",
  windowsHide: true,
});

if (metadata.error || metadata.status !== 0) {
  console.error("Tauri起動前のCargo確認に失敗しました。");
  console.error(`検出したCargo: ${toolchainCargo ?? join(cargoBins[0], cargoExecutable)}`);
  console.error(`PATH先頭: ${missingCargoBins.join(delimiter)}`);
  console.error(metadata.error?.message ?? metadata.stderr.trim());
  process.exit(1);
}

console.log(`[tauri-launcher] Cargo: ${toolchainCargo ?? join(cargoBins[0], cargoExecutable)}`);

const child = spawn(process.execPath, [tauriCli, ...process.argv.slice(2)], {
  cwd: projectDirectory,
  env,
  stdio: "inherit",
});

child.on("error", error => {
  console.error(`Tauri CLIを起動できませんでした: ${error.message}`);
  process.exitCode = 1;
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exitCode = code ?? 1;
});
