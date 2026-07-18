import test, { after } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "vite";

const vite = await createServer({
  logLevel: "silent",
  server: { middlewareMode: true },
  appType: "custom",
});
after(() => vite.close());

const {
  persistenceCommandAfterWorkspaceEdit,
  restoredWorkspaceStep,
  workspaceArtifactReadiness,
} = await vite.ssrLoadModule(
  "/src/workspace/phaseModel.ts",
);

test("initial STEP4/5 saves use the forward-only progress command", () => {
  assert.equal(
    persistenceCommandAfterWorkspaceEdit(4, 5),
    "update_expression_workspace_step",
  );
  assert.equal(
    persistenceCommandAfterWorkspaceEdit(5, 6),
    "update_expression_workspace_step",
  );
});

test("completed work explicitly regresses only after a STEP4/5 save", () => {
  assert.equal(
    persistenceCommandAfterWorkspaceEdit(7, 5),
    "regress_expression_workspace_step",
  );
  assert.equal(
    persistenceCommandAfterWorkspaceEdit(7, 6),
    "regress_expression_workspace_step",
  );
});

test("dirty or stale materials lock every downstream artifact", () => {
  const dirty = workspaceArtifactReadiness({
    currentStep: 7,
    requestDirty: true,
    generatedReady: true,
    extractedComplete: true,
    basePreviewReady: true,
    rifeCurrent: true,
  });
  assert.deepEqual(dirty, {
    step2Ready: false,
    step3Complete: false,
    step4Complete: false,
    step5Complete: false,
    step6Complete: false,
  });

  const checkpoint3 = workspaceArtifactReadiness({
    currentStep: 3,
    requestDirty: false,
    generatedReady: true,
    extractedComplete: true,
    basePreviewReady: true,
    rifeCurrent: true,
  });
  assert.equal(checkpoint3.step2Ready, true);
  assert.equal(checkpoint3.step3Complete, false);
  assert.equal(checkpoint3.step6Complete, false);
});

test("resume never promotes a checkpoint from leftover files", () => {
  const everyArtifact = {
    generatedReady: true,
    extractedReady: true,
    compositeReady: true,
    rifeReady: true,
  };
  assert.equal(restoredWorkspaceStep(2, everyArtifact), 2);
  assert.equal(restoredWorkspaceStep(3, everyArtifact), 3);
  assert.equal(restoredWorkspaceStep(6, everyArtifact), 6);
  assert.equal(restoredWorkspaceStep(7, everyArtifact), 7);
});

test("resume falls back when the checkpoint artifact is missing", () => {
  assert.equal(restoredWorkspaceStep(7, {
    generatedReady: true,
    extractedReady: true,
    compositeReady: true,
    rifeReady: false,
  }), 6);
  assert.equal(restoredWorkspaceStep(6, {
    generatedReady: true,
    extractedReady: true,
    compositeReady: false,
    rifeReady: false,
  }), 4);
  assert.equal(restoredWorkspaceStep(4, {
    generatedReady: true,
    extractedReady: false,
    compositeReady: false,
    rifeReady: false,
  }), 3);
  assert.equal(restoredWorkspaceStep(3, {
    generatedReady: false,
    extractedReady: false,
    compositeReady: false,
    rifeReady: false,
  }), 2);
});
