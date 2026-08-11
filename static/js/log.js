const MAX_ENTRIES = 50;

export function logMessage(text, level = 'info') {
    const list = document.getElementById('log-list');
    if (!list) return;

    const time = new Date().toLocaleTimeString([], { hour12: false });
    const entry = document.createElement('li');
    entry.className = `log-entry log-${level}`;
    entry.innerHTML = `<span class="log-time">${time}</span>${text}`;
    list.appendChild(entry);

    while (list.children.length > MAX_ENTRIES) {
        list.removeChild(list.firstChild);
    }
    list.scrollTop = list.scrollHeight;
}
