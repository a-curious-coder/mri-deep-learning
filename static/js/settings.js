import { logMessage } from './log.js';

export async function updateSettings() {
    const settings = {
        image_size: document.getElementById('image_size').value,
        slice_mode: document.getElementById('slice_mode').value,
        test_size: document.getElementById('test_size').value,
        val_size: document.getElementById('val_size').value,
    };

    try {
        const response = await fetch('/update_settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(settings),
        });

        if (response.ok) {
            logMessage('Settings updated successfully', 'success');
        } else {
            logMessage('Failed to update settings', 'error');
        }
    } catch (error) {
        logMessage(`Settings request failed: ${error.message}`, 'error');
    }
}
