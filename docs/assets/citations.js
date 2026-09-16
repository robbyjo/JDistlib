(function () {
  'use strict';

  var input = document.getElementById('citation-search');
  var count = document.getElementById('citation-count');
  var empty = document.getElementById('citation-empty');
  if (!input || !count || !empty) return;

  var entries = Array.prototype.slice.call(document.querySelectorAll('.citation-entry'));
  var groups = Array.prototype.slice.call(document.querySelectorAll('.citation-group'));

  function normalize(value) {
    return value.toLocaleLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
  }

  function filter() {
    var query = normalize(input.value.trim());
    var terms = query.split(/\s+/).filter(Boolean);
    var visible = 0;

    entries.forEach(function (entry) {
      var haystack = normalize(entry.textContent + ' ' + (entry.getAttribute('data-keywords') || ''));
      var match = terms.every(function (term) { return haystack.indexOf(term) !== -1; });
      entry.hidden = !match;
      if (match) visible++;
    });

    groups.forEach(function (group) {
      group.hidden = !group.querySelector('.citation-entry:not([hidden])');
    });

    count.textContent = visible + (visible === 1 ? ' citation' : ' citations') + ' shown';
    empty.hidden = visible !== 0;
  }

  input.addEventListener('input', filter);
  filter();
}());
