#!/usr/bin/env node
/**
 * Export presentation slides to a single PDF.
 * Usage: node export-pdf.mjs
 * Requires: playwright (npm install --save-dev playwright)
 *
 * Serves files via a local HTTP server so <object> SVGs load correctly.
 */
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import http from 'http';
import fs from 'fs';
import path from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outputPath = join(__dirname, 'presentation.pdf');

const SLIDE_WIDTH = 1440;
const SLIDE_HEIGHT = 810;
const TOTAL_SLIDES = 11;
const SCALE = 2; // retina quality

// Minimal static file server
function startServer(root, port) {
  const mimeTypes = {
    '.html': 'text/html', '.css': 'text/css', '.js': 'application/javascript',
    '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.woff2': 'font/woff2',
    '.woff': 'font/woff', '.ttf': 'font/ttf',
  };
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const filePath = join(root, decodeURIComponent(req.url.split('?')[0]));
      const ext = path.extname(filePath);
      fs.readFile(filePath, (err, data) => {
        if (err) { res.writeHead(404); res.end(); return; }
        res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
        res.end(data);
      });
    });
    server.listen(port, () => resolve(server));
  });
}

async function exportPDF() {
  const PORT = 9876;
  const server = await startServer(join(__dirname, '..'), PORT);
  console.log(`  Server running on http://localhost:${PORT}`);

  const browser = await chromium.launch({ channel: 'chromium' });
  const page = await browser.newPage({
    viewport: { width: SLIDE_WIDTH, height: SLIDE_HEIGHT },
    deviceScaleFactor: SCALE,
  });

  await page.goto(`http://localhost:${PORT}/slides/presentation.html`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);

  // Disable stage scaling, move nav/progress inside stage
  await page.evaluate(() => {
    const stage = document.getElementById('stage');
    stage.style.transform = 'none';

    const nav = document.querySelector('.nav-bar');
    const prog = document.querySelector('.progress-bar');
    if (nav) { nav.style.position = 'absolute'; stage.appendChild(nav); }
    if (prog) { prog.style.position = 'absolute'; stage.appendChild(prog); }

    // Hide decorative canvas backgrounds
    document.querySelectorAll('canvas').forEach(c => c.style.display = 'none');

    // Force all animations to end state
    document.querySelectorAll('.anim').forEach(el => {
      el.style.opacity = '1';
      el.style.transform = 'none';
      el.style.transition = 'none';
      el.style.animationDelay = '0s';
      el.style.animationDuration = '0s';
    });

    // Force inline trajectory lines
    document.querySelectorAll('.traj-line').forEach(el => {
      el.style.strokeDashoffset = '0';
      el.style.animation = 'none';
    });
    document.querySelectorAll('.dot-pulse').forEach(el => {
      el.style.animation = 'none';
    });
  });

  await page.waitForTimeout(500);

  // Briefly show all slides so <object> SVGs load
  await page.evaluate(() => {
    document.querySelectorAll('.slide').forEach(s => {
      s.style.display = 'flex';
      s.style.opacity = '0';
      s.style.pointerEvents = 'none';
    });
  });
  await page.waitForTimeout(1500);

  // Inject animation overrides into all SVG objects
  await page.evaluate(() => {
    document.querySelectorAll('object[data$=".svg"]').forEach(obj => {
      const svgDoc = obj.contentDocument;
      if (!svgDoc) return;
      const style = svgDoc.createElementNS('http://www.w3.org/2000/svg', 'style');
      style.textContent = `
        *, *::before, *::after {
          animation: none !important;
          transition: none !important;
        }
        .traj-line {
          stroke-dashoffset: 0 !important;
        }
      `;
      svgDoc.documentElement.prepend(style);
    });

    // Restore slide visibility
    document.querySelectorAll('.slide').forEach(s => {
      s.style.display = '';
      s.style.opacity = '';
      s.style.pointerEvents = '';
    });
  });

  await page.waitForTimeout(300);

  // Screenshot each slide
  const screenshots = [];
  for (let i = 1; i <= TOTAL_SLIDES; i++) {
    await page.evaluate((n) => {
      goTo(n);
      document.querySelectorAll('.slide.active .anim').forEach(el => {
        el.style.opacity = '1';
        el.style.transform = 'none';
        el.style.transition = 'none';
      });
    }, i);
    await page.waitForTimeout(400);
    const buf = await page.screenshot({
      clip: { x: 0, y: 0, width: SLIDE_WIDTH, height: SLIDE_HEIGHT },
    });
    screenshots.push(buf);
    console.log(`  Captured slide ${i}/${TOTAL_SLIDES}`);
  }

  // Assemble PDF — use scaled pixel dimensions so images render 1:1
  const pxW = SLIDE_WIDTH * SCALE;
  const pxH = SLIDE_HEIGHT * SCALE;

  const imgPage = await browser.newPage({
    viewport: { width: pxW, height: pxH },
  });

  const imgTags = screenshots
    .map((buf) => {
      const b64 = buf.toString('base64');
      return `<img src="data:image/png;base64,${b64}" style="width:${pxW}px;height:${pxH}px;display:block;">`;
    })
    .join('');

  const tmpHtml = join(__dirname, '_pdf-assembly.html');
  const htmlContent = `<html>
    <head>
    <title>presentation</title>
    <style>
      * { margin: 0; padding: 0; }
      @page { size: ${pxW}px ${pxH}px; margin: 0; }
      body { width: ${pxW}px; }
      img { page-break-after: always; }
      img:last-child { page-break-after: avoid; }
    </style></head>
    <body>${imgTags}</body>
    </html>`;
  fs.writeFileSync(tmpHtml, htmlContent);
  await imgPage.goto(`http://localhost:${PORT}/slides/_pdf-assembly.html`, { waitUntil: 'load' });

  await imgPage.pdf({
    path: outputPath,
    width: `${pxW}px`,
    height: `${pxH}px`,
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
    printBackground: true,
  });

  fs.unlinkSync(tmpHtml);

  console.log(`\n  PDF saved to: ${outputPath}`);
  await browser.close();
  server.close();
}

exportPDF().catch((err) => {
  console.error(err);
  process.exit(1);
});
