/* Self-test: open index.html#selftest to mount every demo at once and list the errors in #selftest-result. */
(function () {
  const TT = window.TT;
  const errors = [];
  window.addEventListener('error', (e) => errors.push('window error: ' + e.message + ' @' + (e.filename || '').split('/').pop() + ':' + e.lineno));
  document.addEventListener('DOMContentLoaded', () => {
    const only = /^#only=(.+)$/.exec(location.hash);
    if (only) { document.querySelectorAll('[data-demo]').forEach(d => { if (d.dataset.demo !== only[1]) d.remove(); }); document.querySelector('nav.toc').remove(); document.querySelectorAll('main > section > :not([data-demo])').forEach(x => x.remove()); setTimeout(() => TT.mountAll(), 200); return; }
    if (location.hash !== '#selftest') return;
    setTimeout(() => {
      TT.mountAll();
      setTimeout(() => {
        const demos = [...document.querySelectorAll('[data-demo]')];
        const bad = demos.filter(d => d.querySelector('.bad-t') && /demo error|missing demo/.test(d.textContent));
        const names = demos.map(d => d.dataset.demo + (d.dataset.mounted ? '' : ' (not mounted)'));
        const pre = document.createElement('pre'); pre.id = 'selftest-result';
        pre.textContent = `SELFTEST demos=${demos.length} bad=${bad.length} errors=${errors.length}\n` + bad.map(d => d.dataset.demo + ': ' + d.textContent.slice(-160)).join('\n') + '\n' + errors.join('\n') + '\nunmounted: ' + names.filter(n => n.includes('not mounted')).join(', ');
        document.body.prepend(pre);
      }, 600);
    }, 300);
  });
})();
