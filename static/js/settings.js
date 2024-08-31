export async function updateSettings() {
    const settings = {
        image_size: document.getElementById('image_size').value,
        slice_mode: document.getElementById('slice_mode').value,
        test_size: document.getElementById('test_size').value,
        val_size: document.getElementById('val_size').value,
    };

    const response = await fetch('/update_settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(settings),
    });

    if (response.ok) {
        alert('Settings updated successfully');
    } else {
        alert('Failed to update settings');
    }
}
