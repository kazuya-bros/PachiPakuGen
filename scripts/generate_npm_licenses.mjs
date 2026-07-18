import { spawnSync } from "node:child_process";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const projectRoot = resolve(import.meta.dirname, "..");
const outputPath = resolve(projectRoot, process.argv[2] ?? "licenses/NPM_DEPENDENCIES.html");
const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
const npmArgs = ["ls", "--omit=dev", "--all", "--json", "--long"];
const command = process.env.npm_execpath ? process.execPath : npmCommand;
const commandArgs = process.env.npm_execpath ? [process.env.npm_execpath, ...npmArgs] : npmArgs;
const result = spawnSync(command, commandArgs, {
  cwd: projectRoot,
  encoding: "utf8",
  shell: !process.env.npm_execpath && process.platform === "win32",
});

if (result.status !== 0) {
  throw new Error(`npm dependency tree could not be read:\n${result.stderr || result.stdout}`);
}

const tree = JSON.parse(result.stdout);
const packages = new Map();

function visit(node, isRoot = false) {
  if (!isRoot && node.name && node.version && node.path) {
    const key = `${node.name}@${node.version}`;
    if (!packages.has(key)) packages.set(key, node);
  }
  for (const dependency of Object.values(node.dependencies ?? {})) visit(dependency);
}

visit(tree, true);

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function repositoryUrl(packageJson) {
  const repository = packageJson.repository;
  const raw = typeof repository === "string" ? repository : repository?.url;
  return raw?.replace(/^git\+/, "").replace(/\.git$/, "") ?? packageJson.homepage ?? "";
}

const entries = [...packages.values()]
  .map(node => {
    const packageJson = JSON.parse(readFileSync(resolve(node.path, "package.json"), "utf8"));
    const licenseFiles = readdirSync(node.path, { withFileTypes: true })
      .filter(entry => entry.isFile() && /^(licen[sc]e|copying|notice)([._-].*)?$/i.test(entry.name))
      .map(entry => ({
        name: entry.name,
        text: readFileSync(resolve(node.path, entry.name), "utf8").trim(),
      }));
    if (licenseFiles.length === 0) {
      throw new Error(`No license file found for ${node.name}@${node.version} (${node.path})`);
    }
    return {
      name: node.name,
      version: node.version,
      license: packageJson.license ?? node.license ?? "UNKNOWN",
      repository: repositoryUrl(packageJson),
      licenseFiles,
    };
  })
  .sort((a, b) => a.name.localeCompare(b.name) || a.version.localeCompare(b.version));

const sections = entries.map(entry => {
  const title = `${entry.name} ${entry.version}`;
  const link = entry.repository
    ? `<a href="${escapeHtml(entry.repository)}">${escapeHtml(title)}</a>`
    : escapeHtml(title);
  const texts = entry.licenseFiles
    .map(file => `<h3>${escapeHtml(file.name)}</h3><pre>${escapeHtml(file.text)}</pre>`)
    .join("\n");
  return `<section><h2>${link}</h2><p>SPDX: <code>${escapeHtml(entry.license)}</code></p>${texts}</section>`;
});

const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PachiPakuGen JavaScript Dependency Licenses</title>
  <style>
    body { font-family: sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
    section { border-top: 1px solid #bbb; padding: 1rem 0; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f4f6f8; padding: 1rem; }
  </style>
</head>
<body>
  <h1>PachiPakuGen JavaScript Dependency Licenses</h1>
  <p>Generated from the production dependency tree locked by package-lock.json. Development-only build tools are excluded.</p>
  ${sections.join("\n")}
</body>
</html>
`;

writeFileSync(outputPath, html, "utf8");
console.log(`Wrote ${entries.length} production dependency notices to ${outputPath}`);
