document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const searchBtnText = searchBtn.querySelector('span');
    const loader = searchBtn.querySelector('.loader');
    const modeBtns = document.querySelectorAll('.toggle-btn');
    const resultsContainer = document.getElementById('resultsContainer');

    let currentMode = 'cross';

    // Handle Mode Toggle
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;
            searchInput.focus();
        });
    });

    // Handle Enter Key
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch();
        }
    });

    // Handle Search Button Click
    searchBtn.addEventListener('click', performSearch);

    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        // UI Loading State
        setLoading(true);

        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    mode: currentMode,
                    top_k: 5
                })
            });

            if (!response.ok) throw new Error('Search failed');

            const data = await response.json();
            renderResults(data.results);
            
        } catch (error) {
            console.error(error);
            resultsContainer.innerHTML = `
                <div class="placeholder-state" style="color: #ef4444; border-color: rgba(239, 68, 68, 0.2);">
                    <p>An error occurred while searching. Please ensure the backend is running.</p>
                </div>
            `;
        } finally {
            setLoading(false);
        }
    }

    function setLoading(isLoading) {
        if (isLoading) {
            searchBtnText.classList.add('hidden');
            loader.classList.remove('hidden');
            searchInput.disabled = true;
            modeBtns.forEach(b => b.style.pointerEvents = 'none');
        } else {
            searchBtnText.classList.remove('hidden');
            loader.classList.add('hidden');
            searchInput.disabled = false;
            searchInput.focus();
            modeBtns.forEach(b => b.style.pointerEvents = 'auto');
        }
    }

    function renderResults(results) {
        resultsContainer.innerHTML = '';

        if (!results || results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="placeholder-state">
                    <p>No relevant fact-checks found.</p>
                </div>
            `;
            return;
        }

        results.forEach((result, index) => {
            const card = document.createElement('div');
            card.className = 'result-card glass';
            // Stagger animation delays for a cascading effect
            card.style.animationDelay = `${index * 0.1}s`;

            card.innerHTML = `
                <div class="card-header">
                    <span class="rank-badge">Rank ${result.rank}</span>
                    <span class="id-badge">ID: ${result.id}</span>
                </div>
                <div class="result-content">
                    <p class="result-text">${result.text}</p>
                </div>
            `;
            
            resultsContainer.appendChild(card);
        });
    }
});
