/**
 * Main JavaScript file for AI Document Chat
 */

document.addEventListener('DOMContentLoaded', function() {
    
    // File upload validation
    const fileInput = document.getElementById('file');
    const uploadForm = document.getElementById('upload-form');
    
    if (fileInput && uploadForm) {
        fileInput.addEventListener('change', function() {
            validateFile(this);
        });
        
        uploadForm.addEventListener('submit', function(e) {
            if (!validateFile(fileInput)) {
                e.preventDefault();
            }
        });
    }
    
    /**
     * Validate uploaded file
     * @param {HTMLInputElement} fileInput - The file input element
     * @returns {boolean} - Whether the file is valid
     */
    function validateFile(fileInput) {
        if (!fileInput.files || fileInput.files.length === 0) {
            showError('Please select a file to upload.');
            return false;
        }
        
        const file = fileInput.files[0];
        const allowedTypes = ['application/pdf', 'application/epub+zip', 'text/plain', 
                            'application/x-mobipocket-ebook', 'application/vnd.amazon.mobi8-ebook'];
        const allowedExtensions = ['pdf', 'epub', 'txt', 'mobi', 'azw', 'azw3'];
        
        // Check file size (20MB maximum)
        const maxSize = 20 * 1024 * 1024; // 20MB in bytes
        if (file.size > maxSize) {
            showError(`File is too large. Maximum size is 20MB.`);
            return false;
        }
        
        // Check file extension
        const fileName = file.name;
        const fileExtension = fileName.split('.').pop().toLowerCase();
        if (!allowedExtensions.includes(fileExtension)) {
            showError(`File type not allowed. Supported formats: ${allowedExtensions.join(', ')}`);
            return false;
        }
        
        return true;
    }
    
    /**
     * Show an error message
     * @param {string} message - The error message to display
     */
    function showError(message) {
        // Create alert element or use an existing container
        const alertContainer = document.createElement('div');
        alertContainer.className = 'alert alert-danger alert-dismissible fade show mt-2';
        alertContainer.role = 'alert';
        alertContainer.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Insert before the form
        if (uploadForm) {
            uploadForm.parentNode.insertBefore(alertContainer, uploadForm);
        }
        
        // Auto dismiss after 5 seconds
        setTimeout(() => {
            alertContainer.classList.remove('show');
            setTimeout(() => alertContainer.remove(), 150);
        }, 5000);
    }
});
