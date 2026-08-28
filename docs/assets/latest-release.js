(function () {
  'use strict';

  function showVersion(version) {
    document.querySelectorAll('[data-latest-release-version]').forEach(function (element) {
      element.textContent = version;
    });
  }

  fetch('https://api.github.com/repos/robbyjo/JDistlib/releases/latest', {
    headers: { Accept: 'application/vnd.github+json' }
  }).then(function (response) {
    if (!response.ok) throw new Error('GitHub release lookup failed');
    return response.json();
  }).then(function (release) {
    var match = /^v(\d+\.\d+\.\d+(?:[-.][0-9A-Za-z.-]+)?)$/.exec(String(release.tag_name || ''));
    if (match) showVersion(match[1]);
  }).catch(function () {
    // The HTML contains the current release as a no-network/rate-limit fallback.
  });
}());
