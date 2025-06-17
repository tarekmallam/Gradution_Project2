document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.querySelector('.upload-btn');
    const previewSection = document.querySelector('.preview-section');
    const previewImage = document.getElementById('previewImage');
    const analyzeBtn = document.querySelector('.analyze-btn');
    const results = document.querySelector('.results');
    const loading = document.querySelector('.loading');
    const classification = document.getElementById('classification');
    const confidence = document.getElementById('confidence');
    const errorMsg = document.getElementById('errorMsg');

    let selectedFile = null;

    uploadBtn.addEventListener('click', () => {
        fileInput.click();
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#3498db';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#dcdde1';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#dcdde1';
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewSection.style.display = 'block';
                    results.style.display = 'none';
                    errorMsg.textContent = '';
                };
                reader.readAsDataURL(file);
            } else {
                errorMsg.textContent = 'Please upload an image file.';
            }
        }
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) {
            errorMsg.textContent = 'Please upload an image first.';
            return;
        }

        loading.style.display = 'block';
        analyzeBtn.disabled = true;
        errorMsg.textContent = '';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (response.ok) {
                classification.textContent = data.class;
                confidence.textContent = data.confidence.toFixed(1);
                results.style.display = 'block';
            } else {
                errorMsg.textContent = data.error || 'An error occurred during analysis.';
                results.style.display = 'none';
            }
        } catch (error) {
            errorMsg.textContent = 'An error occurred during analysis.';
            results.style.display = 'none';
            console.error('Error:', error);
        } finally {
            loading.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });
});