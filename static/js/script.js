$(document).ready(function() {
    const uploadArea = $('.upload-container');
    const fileInput = $('#file-input');
    const processingIndicator = $('.processing-container');
    const resultsArea = $('.results-container');

    let formData; // Declare formData in a broader scope
    let startX, startY, endX, endY;
    let isDrawing = false;

    // Handle drag and drop
    uploadArea.on('dragover', function(e) {
        e.preventDefault();
        $(this).addClass('dragover');
    });

    uploadArea.on('dragleave', function(e) {
        e.preventDefault();
        $(this).removeClass('dragover');
    });

    uploadArea.on('drop', function(e) {
        e.preventDefault();
        $(this).removeClass('dragover');
        const files = e.originalEvent.dataTransfer.files;
        handleFile(files[0]);
    });

    // Handle click upload
    uploadArea.on('click', function() {
        fileInput.click();
    });

    fileInput.on('change', function() {
        handleFile(this.files[0]);
    });

    function handleFile(file) {
        if (file) {
            processingIndicator.html(`
                <div class="dot-spinner">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
                <p class="processing-text">Processing</p>
            `);
            processingIndicator.show();
            uploadArea.addClass('minimized');
            
            formData = new FormData();
            formData.append('file', file);

            $.ajax({
                url: '/',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function(response) {
                    processingIndicator.hide();
                    
                    if (response.warning) {
                        handlePSNRWarning(response);
                    } else {
                        showResults(response);
                    }
                },
                error: function(xhr, status, error) {
                    processingIndicator.hide();
                    uploadArea.removeClass('minimized');
                    console.error('Error:', error);
                    alert('Error processing image. Please try again.');
                }
            });
        }
    }

    function showResults(response) {
        // Show the Go Back button
        $('#go-back').show();
        
        // Update images and metrics
        $('#original-image').attr('src', '/uploads/' + response.original);
        $('#deblurred-image').attr('src', '/results/' + response.result);
        
        // Update metrics and download button
        $('.result-info').html(`
            <p>PSNR: ${response.psnr.toFixed(4)}</p>
            <p>SSIM: ${response.ssim.toFixed(4)}</p>
            <button class="download-btn">Download</button>
        `);

        // Add click handler for download button
        $('.result-info .download-btn').on('click', function() {
            window.location.href = '/download/' + response.result;
        });
        
        // Add action buttons if they don't exist
        if ($('#deblur-selection').length === 0) {
            const actionButtons = `
                <div class="button-container">
                    <button id="deblur-selection" class="action-button">Deblur Region</button>
                    <button id="compare-results-btn" class="action-button">Compare Results</button>
                </div>
            `;
            resultsArea.append(actionButtons);
            
            // Add click handlers for the buttons
            $('#deblur-selection').on('click', handleDeblurSelection);
            $('#compare-results-btn').on('click', compareResults);
        }
        
        // Show results area but keep upload area visible
        resultsArea.show();
        uploadArea.addClass('minimized');
        
        // Initialize canvas for region selection
        initializeCanvas();
    }

    function updateRegionResult(response) {
        $('#region-result').show().find('.panel-content').html(`
            <img src="${response.result}" alt="Deblurred Region">
            <div class="result-info">
                <p>PSNR: ${response.psnr.toFixed(4)}</p>
                <p>SSIM: ${response.ssim.toFixed(4)}</p>
                <button class="download-btn" onclick="window.location.href='/download/${response.result.split('/').pop()}'">Download</button>
            </div>
        `);
    }

    function getBetterMethod(entirePSNR, entireSSIM, regionPSNR, regionSSIM) {
        const psnrDifference = regionPSNR - entirePSNR;
        const ssimDifference = regionSSIM - entireSSIM;
        
        if (Math.abs(psnrDifference) < 0.5 && Math.abs(ssimDifference) < 0.05) {
            return 'Both methods perform similarly. The difference is not significant.';
        } else if (psnrDifference > 0.5 && ssimDifference > 0.05) {
            return 'Region deblurring performs better in both metrics.';
        } else if (psnrDifference < -0.5 && ssimDifference < -0.05) {
            return 'Entire image deblurring performs better in both metrics.';
        } else if (psnrDifference < -0.5 && ssimDifference > 0.05) {
            if (ssimDifference > 0.1) {
                return 'Although region deblurring has lower PSNR, it performs significantly better in SSIM, suggesting better structural preservation.';
            } else {
                return 'Region deblurring has lower PSNR but slightly better SSIM. The improvement in structural similarity is not significant.';
            }
        } else if (psnrDifference > 0.5 && ssimDifference < -0.05) {
            if (Math.abs(ssimDifference) > 0.1) {
                return 'Region deblurring has better PSNR but significantly lower SSIM. Consider using entire image deblurring for better structural preservation.';
            } else {
                return 'Region deblurring has better PSNR but slightly lower SSIM. The difference in structural similarity is not significant.';
            }
        } else {
            return 'Results are mixed with minimal differences. Both methods perform similarly overall.';
        }
    }

    function initializeCanvas() {
        const canvas = document.getElementById('selection-canvas');
        const img = document.getElementById('original-image');
        
        if (!canvas || !img) return;
        
        const ctx = canvas.getContext('2d');
        
        // Set canvas size to match image
        function resizeCanvas() {
            canvas.width = img.width;
            canvas.height = img.height;
            canvas.style.width = img.width + 'px';
            canvas.style.height = img.height + 'px';
        }
        
        img.onload = resizeCanvas;
        resizeCanvas();

        // Mouse events for selection
        canvas.addEventListener('mousedown', function(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            
            startX = (e.clientX - rect.left) * scaleX;
            startY = (e.clientY - rect.top) * scaleY;
            isDrawing = true;
            
            // Clear previous selection
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        });

        canvas.addEventListener('mousemove', function(e) {
            if (!isDrawing) return;
            
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            
            endX = (e.clientX - rect.left) * scaleX;
            endY = (e.clientY - rect.top) * scaleY;
            
            // Clear and redraw
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = '#FF0000';
            ctx.lineWidth = 2;
            ctx.strokeRect(
                Math.min(startX, endX),
                Math.min(startY, endY),
                Math.abs(endX - startX),
                Math.abs(endY - startY)
            );
        });

        canvas.addEventListener('mouseup', function() {
            isDrawing = false;
        });

        canvas.addEventListener('mouseleave', function() {
            if (isDrawing) {
                isDrawing = false;
            }
        });
    }

    function handleDeblurSelection() {
        if (!startX || !startY || !endX || !endY) {
            alert('Please select a region first by dragging on the image.');
            return;
        }

        // Show the region processing spinner
        $('#region-result').html(`
            <div class="image-panel">
                <div class="panel-header">Deblurred Selected Region</div>
                <div class="panel-content">
                    <div class="region-processing-container">
                        <div class="region-dot-spinner">
                            <div class="region-dot"></div>
                            <div class="region-dot"></div>
                            <div class="region-dot"></div>
                        </div>
                        <p class="region-processing-text">Processing region...</p>
                    </div>
                </div>
            </div>
        `).show();

        const canvas = document.getElementById('selection-canvas');
        const img = document.getElementById('original-image');
        
        // Get the scale factors
        const scaleX = img.naturalWidth / canvas.width;
        const scaleY = img.naturalHeight / canvas.height;

        // Calculate the selected region coordinates
        const regionData = {
            x: Math.round(Math.min(startX, endX) * scaleX),
            y: Math.round(Math.min(startY, endY) * scaleY),
            width: Math.round(Math.abs(endX - startX) * scaleX),
            height: Math.round(Math.abs(endY - startY) * scaleY)
        };

        // Get the original image filename from the src attribute
        const imagePath = img.src;
        const filename = imagePath.split('/').pop();

        // Create form data
        const formData = new FormData();
        formData.append('file', filename);
        formData.append('region', JSON.stringify(regionData));

        // Send AJAX request
        $.ajax({
            url: '/deblur_region',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.warning) {
                    // Show warning modal for low PSNR
                    const modal = $('#psnr-warning-modal');
                    const modalContent = modal.find('.modal-content p');
                    modalContent.text("The selected region has low PSNR. Do you want to continue with deblurring?");
                    modal.show();

                    $('#yes-deblur').off('click').on('click', function() {
                        modal.hide();
                        updateRegionResult(response);
                    });

                    $('#no-deblur').off('click').on('click', function() {
                        modal.hide();
                        $('#region-result').html(`
                            <div class="image-panel">
                                <div class="panel-header">Deblurred Selected Region</div>
                                <div class="panel-content">
                                    <div class="region-placeholder">
                                        <i class="fas fa-crop"></i>
                                        <p>Select a region to deblur</p>
                                    </div>
                                </div>
                            </div>
                        `);
                    });
                } else {
                    updateRegionResult(response);
                }
            },
            error: function(xhr, status, error) {
                console.error('Error:', error);
                $('#region-result').html(`
                    <div class="image-panel">
                        <div class="panel-header">Deblurred Selected Region</div>
                        <div class="panel-content">
                            <p style="color: red;">Error processing region. Please try again.</p>
                        </div>
                    </div>
                `);
            }
        });
    }

    function compareResults() {
        // Get values from the deblurred image panel
        const entireImagePSNR = parseFloat($('#deblurred-image').siblings('.result-info').find('p:contains("PSNR")').text().replace('PSNR: ', ''));
        const entireImageSSIM = parseFloat($('#deblurred-image').siblings('.result-info').find('p:contains("SSIM")').text().replace('SSIM: ', ''));
        
        // Get values from the region result panel - Fix the selectors
        const selectedRegionPSNR = parseFloat($('#region-result .result-info').find('p:contains("PSNR")').text().replace('PSNR: ', ''));
        const selectedRegionSSIM = parseFloat($('#region-result .result-info').find('p:contains("SSIM")').text().replace('SSIM: ', ''));

        // Debug logging
        console.log('Entire Image PSNR:', entireImagePSNR);
        console.log('Entire Image SSIM:', entireImageSSIM);
        console.log('Region PSNR:', selectedRegionPSNR);
        console.log('Region SSIM:', selectedRegionSSIM);

        const comparisonResult = `
            <div class="comparison-results">
                <h3>Comparison Results</h3>
                <table>
                    <tr>
                        <th>Image</th>
                        <th>PSNR</th>
                        <th>SSIM</th>
                    </tr>
                    <tr>
                        <td>Deblurred Entire Image</td>
                        <td>${!isNaN(entireImagePSNR) ? entireImagePSNR.toFixed(4) : 'N/A'}</td>
                        <td>${!isNaN(entireImageSSIM) ? entireImageSSIM.toFixed(4) : 'N/A'}</td>
                    </tr>
                    <tr>
                        <td>Deblurred Selected Region</td>
                        <td>${!isNaN(selectedRegionPSNR) ? selectedRegionPSNR.toFixed(4) : 'N/A'}</td>
                        <td>${!isNaN(selectedRegionSSIM) ? selectedRegionSSIM.toFixed(4) : 'N/A'}</td>
                    </tr>
                </table>
                <p><strong>Analysis:</strong> ${getBetterMethod(entireImagePSNR, entireImageSSIM, selectedRegionPSNR, selectedRegionSSIM)}</p>
                <p>PSNR Difference: ${!isNaN(selectedRegionPSNR - entireImagePSNR) ? (selectedRegionPSNR - entireImagePSNR).toFixed(4) : 'N/A'} dB (decibels)</p>
                <p>SSIM Difference: ${!isNaN(selectedRegionSSIM - entireImageSSIM) ? (selectedRegionSSIM - entireImageSSIM).toFixed(4) : 'N/A'}</p>
            </div>
        `;

        $('#comparison-result').html(comparisonResult).show();
    }

    function handlePSNRWarning(response) {
        const modal = $('#psnr-warning-modal');
        const modalContent = modal.find('.modal-content p');
        
        modalContent.text("The image has low PSNR. Do you want to continue with deblurring?");
        modal.show();
        
        $('#yes-deblur').off('click').on('click', function() {
            modal.hide();
            $('#go-back').show();
            showResults(response);
        });
        
        $('#no-deblur').off('click').on('click', function() {
            modal.hide();
            $('#go-back').hide();
            // Reset and show upload area
            fileInput.val('');
            uploadArea.show();
            resultsArea.hide();
        });
    }

    // Add Go Back button functionality
    $('#go-back').on('click', function() {
        // Hide the button
        $(this).hide();
        
        // Reset the form and clear results
        $('#file-input').val('');
        $('.results-container').hide();
        $('#region-result').hide();
        $('#comparison-result').hide();
        $('.upload-container').removeClass('minimized');
        
        // Clear any canvas selections
        const canvas = document.getElementById('selection-canvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        
        // Reset any processing states
        $('.processing-container').hide();
        
        // Reset variables
        startX = startY = endX = endY = null;
        isDrawing = false;
    });
});