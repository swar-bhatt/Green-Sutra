document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('adviceForm');
    const submitBtn = document.getElementById('submitBtn');
    const resetBtn = document.getElementById('resetBtn');

    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner"></i> Loading...';

        try {
            const formData = new FormData(form);
            const response = await fetch('/get_advice', {
                method: 'POST', // Ensure POST method
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.text();
                throw new Error(errorData.detail || 'Failed to fetch advice');
            }

            // Since the backend returns an HTML response (result.html),
            // we can set the document's HTML to the response
            const html = await response.text();
            document.open();
            document.write(html);
            document.close();
        } catch (error) {
            alert(`Error: ${error.message}. Please try again later.`);
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-check"></i> Get Advice';
        }
    });

    // Handle reset button
    resetBtn.addEventListener('click', () => {
        form.reset();
    });

});