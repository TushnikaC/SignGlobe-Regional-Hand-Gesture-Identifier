document.getElementById('toggle-dropdown').addEventListener('click', () => {
    const dropdown = document.getElementById('region-select');
    if (dropdown.classList.contains('hidden')) {
        dropdown.classList.remove('hidden');
        dropdown.style.display = 'block';
    } else {
        dropdown.classList.add('hidden');
        dropdown.style.display = 'none';
    }
});

document.getElementById('start-capturing').addEventListener('click', () => {
    const selectedRegions = Array.from(
        document.getElementById('region-select').selectedOptions
    ).map(option => option.value);

    if (selectedRegions.length > 0) {
        alert(`Starting capture for regions: ${selectedRegions.join(', ')}`);
        document.getElementById('video-section').classList.remove('hidden');
    } else {
        alert('Please select at least one region!');
    }
});
// Toggle Dropdown Visibility
document.getElementById('selectRegionsButton').addEventListener('click', () => {
    const dropdown = document.getElementById('regionDropdown');
    dropdown.classList.toggle('active'); // Toggle the "active" class
});
