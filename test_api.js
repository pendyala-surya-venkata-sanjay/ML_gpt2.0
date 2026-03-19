// Test API integration from frontend perspective
const API_BASE_URL = 'http://127.0.0.1:8000';

async function testAPI() {
    console.log('Testing API integration...');
    
    try {
        // Test root endpoint
        const rootResponse = await fetch(`${API_BASE_URL}/`);
        console.log('Root status:', rootResponse.status);
        const rootData = await rootResponse.json();
        console.log('Root data:', rootData);
        
        // Test upload endpoint
        const formData = new FormData();
        // Create a simple CSV file for testing
        const csvContent = 'age,salary,experience\n25,50000,2\n30,60000,5';
        const blob = new Blob([csvContent], { type: 'text/csv' });
        formData.append('file', blob, 'test.csv');
        
        const uploadResponse = await fetch(`${API_BASE_URL}/upload_dataset`, {
            method: 'POST',
            body: formData
        });
        console.log('Upload status:', uploadResponse.status);
        const uploadData = await uploadResponse.json();
        console.log('Upload data:', uploadData);
        
        if (uploadData.status === 'success') {
            // Test training endpoint
            const datasetPath = encodeURIComponent(uploadData.dataset_path);
            const trainResponse = await fetch(`${API_BASE_URL}/train_model?dataset_path=${datasetPath}`, {
                method: 'POST'
            });
            console.log('Training status:', trainResponse.status);
            const trainData = await trainResponse.json();
            console.log('Training data:', trainData);
        }
        
    } catch (error) {
        console.error('API Test Error:', error);
    }
}

testAPI();
