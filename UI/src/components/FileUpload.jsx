import React, { useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { Button } from '@chakra-ui/react';

const FileUpload = () => {
  const fileInputRef = useRef(null);

  const onDrop = useCallback(acceptedFiles => {
    const file = acceptedFiles[0];
    const formData = new FormData();
    formData.append('file', file);

    axios.post('http://localhost:8000/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      responseType: 'blob'  // Ensure the response is handled as a blob
    })
    .then(response => {
      const url = URL.createObjectURL(response.data); // Create a URL from the blob
      const img = document.createElement('img');
      img.src = url;
      document.body.appendChild(img); // Append the image to the body
    })
    .catch(error => {
      console.error('Error uploading file:', error);
    });
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: 'image/jpeg',
    multiple: false,
    noClick: true, // Prevents opening the file dialog on click
    noKeyboard: true, // Prevents opening the file dialog on keyboard input
  });

  const openFileDialog = () => {
    fileInputRef.current.click();
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <div {...getRootProps({ className: 'dropzone' })} style={{ border: '2px dashed gray', padding: '20px' }}>
        <input {...getInputProps()} ref={fileInputRef} style={{ display: 'none' }} />
        <p>Drag 'n' drop a JPEG file here</p>
      </div>
      <Button onClick={openFileDialog} mt={4}>Upload Image</Button>
    </div>
  );
};

export default FileUpload;
