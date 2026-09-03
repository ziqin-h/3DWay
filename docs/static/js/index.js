(() => {
  const copyBtn = document.getElementById("copy-bibtex");
  const bibtex = document.getElementById("bibtex");
  if (copyBtn && bibtex) {
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(bibtex.textContent);
        copyBtn.textContent = "Copied";
        setTimeout(() => {
          copyBtn.textContent = "Copy";
        }, 1600);
      } catch (e) {
        copyBtn.textContent = "Select & copy";
      }
    });
  }

  const frames = Array.from(document.querySelectorAll(".pipeline-frame"));
  if (!frames.length) return;

  const HOLD_MS = 3000;

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const showStatic = (img, src) => {
    img.src = src;
  };

  const playGif = (img) =>
    new Promise((resolve) => {
      const duration = Number(img.dataset.duration) || 10000;
      const gifSrc = `${img.dataset.gif}?t=${Date.now()}`;
      img.src = gifSrc;
      setTimeout(() => {
        showStatic(img, img.dataset.last);
        resolve();
      }, duration);
    });

  const resetAll = () => {
    frames.forEach((img) => showStatic(img, img.dataset.first));
  };

  const runSequence = async () => {
    resetAll();
    for (const img of frames) {
      await playGif(img);
    }
    await sleep(HOLD_MS);
    runSequence();
  };

  runSequence();
})();
