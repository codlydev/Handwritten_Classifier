document.addEventListener('DOMContentLoaded', () => {
    // Get references to DOM elements
    const imageUpload = document.getElementById('imageUpload');
    const convertBtn = document.getElementById('convertBtn');
    const recognizedText = document.getElementById('recognizedText');
    const copyBtn = document.getElementById('copyBtn');

    // Add event listener to the convert button
    convertBtn.addEventListener('click', async () => {
        const file = imageUpload.files[0]; // Get the first selected file
        
        // Check if a file has been selected
        if (!file) {
            showMessageBox('Please select an image first.', 'Error');
            return;
        }

        // Validate file type (must be an image)
        if (!file.type.startsWith('image/')) {
            showMessageBox('Please upload an image file (e.g., JPG, PNG).', 'Error');
            return;
        }

        // Validate file size (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
            showMessageBox('File size exceeds 5MB limit.', 'Error');
            return;
        }

        // Create a FormData object to send the file
        const formData = new FormData();
        formData.append('image', file); // Append the image file with key 'image'

        // Display a converting message and disable button
        recognizedText.value = 'Converting... Please wait.';
        recognizedText.style.color = '#007bff'; // Blue text for loading state
        convertBtn.disabled = true; // Disable button during processing

        try {
            // Send the image file to the backend using fetch API
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            // Re-enable button
            convertBtn.disabled = false;

            // Check if the response was successful
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Something went wrong during conversion.');
            }

            // Parse the JSON response from the backend
            const data = await response.json();
            recognizedText.value = data.text; // Display the recognized text
            recognizedText.style.color = '#333'; // Reset text color
            
            // If the recognized text is empty, inform the user
            if (!data.text || data.text.trim() === '') {
                recognizedText.value = 'No text recognized. Please try with a clearer image.';
                recognizedText.style.color = '#dc3545'; // Red text for no text found
            }

        } catch (error) {
            // Re-enable button on error
            convertBtn.disabled = false;
            console.error('Error:', error);
            recognizedText.value = `Error: ${error.message}`;
            recognizedText.style.color = '#dc3545'; // Red text for error state
            showMessageBox(`Failed to convert text: ${error.message}`, 'Error');
        }
    });

    // Add event listener to the copy button
    copyBtn.addEventListener('click', async () => {
        // Select the text in the textarea
        recognizedText.select();
        recognizedText.setSelectionRange(0, recognizedText.value.length);
        
        try {
            // Try modern Clipboard API first
            if (navigator.clipboard) {
                await navigator.clipboard.writeText(recognizedText.value);
                showMessageBox('Text copied to clipboard!', 'Success');
            } else {
                // Fallback to execCommand for broader compatibility
                const successful = document.execCommand('copy');
                if (successful) {
                    showMessageBox('Text copied to clipboard!', 'Success');
                } else {
                    showMessageBox('Failed to copy text. Please copy manually.', 'Error');
                }
            }
        } catch (err) {
            console.error('Failed to copy text:', err);
            showMessageBox('Failed to copy text. Your browser might not support automatic copying, please copy manually.', 'Error');
        }
    });

    /**
     * Creates and displays a custom message box instead of alert().
     * @param {string} message - The message to display.
     * @param {string} type - The type of message (e.g., 'Success', 'Error', 'Info').
     */
    function showMessageBox(message, type = 'Info') {
        // Remove any existing message box
        const existingMessageBox = document.getElementById('customMessageBox');
        if (existingMessageBox) {
            existingMessageBox.remove();
        }

        const messageBox = document.createElement('div');
        messageBox.id = 'customMessageBox';
        messageBox.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: ${type === 'Error' ? '#dc3545' : type === 'Success' ? '#28a745' : '#007bff'};
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            z-index: 1000;
            font-size: 1.1em;
            text-align: center;
            opacity: 0;
            transition: opacity 0.3s ease-in-out;
            max-width: 80%;
            word-wrap: break-word;
        `;
        messageBox.textContent = message;

        document.body.appendChild(messageBox);

        // Fade in the message box
        setTimeout(() => {
            messageBox.style.opacity = '1';
        }, 10);

        // Fade out and remove after 3 seconds
        setTimeout(() => {
            messageBox.style.opacity = '0';
            messageBox.addEventListener('transitionend', () => messageBox.remove(), { once: true });
        }, 3000);
    }
});