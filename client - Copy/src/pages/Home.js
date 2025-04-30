import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Box,
  Button,
  Container,
  Typography,
  Card,
  CardContent,
  Grid,
  CircularProgress,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

// When running in Docker, API requests will be proxied through Nginx
// When running locally, connect to the backend port directly
const API_URL = window.location.hostname === 'localhost' ? 'http://localhost:5000' : '';

const Home = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [referenceFolders, setReferenceFolders] = useState([]);
  const [selectedFolder, setSelectedFolder] = useState('');

  // Fetch available reference folders on component mount
  useEffect(() => {
    const fetchReferenceFolders = async () => {
      try {
        const response = await axios.get(`${API_URL}/reference-folders`);
        setReferenceFolders(response.data.folders);
        if (response.data.folders.length > 0) {
          setSelectedFolder(response.data.folders[0]);
        }
      } catch (err) {
        console.error('Error fetching reference folders:', err);
      }
    };
    
    fetchReferenceFolders();
  }, []);

  const handleFolderChange = (event) => {
    setSelectedFolder(event.target.value);
  };

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
      setResults(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Convert image to base64
      const reader = new FileReader();
      reader.readAsDataURL(selectedImage);
      reader.onloadend = async () => {
        try {
          const base64Image = reader.result;

          // Upload image to server
          const response = await axios.post(`${API_URL}/api/process_image`, {
            image: base64Image,
            reference_folder: selectedFolder
          });

          if (response.data.error) {
            setError(response.data.error);
          } else {
            setResults(response.data);
          }
        } catch (err) {
          console.error('Error details:', err.response?.data || err.message);
          setError(err.response?.data?.error || 'Error processing image. Please try again.');
        } finally {
          setLoading(false);
        }
      };
    } catch (err) {
      console.error('Error reading file:', err);
      setError('Error reading image file');
      setLoading(false);
    }
  };

  const handleProcess = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Convert image to base64
      const reader = new FileReader();
      reader.readAsDataURL(selectedImage);
      reader.onloadend = async () => {
        try {
          const base64Image = reader.result;

          // Process image for face recognition
          const response = await axios.post(`${API_URL}/api/process_image`, {
            image: base64Image,
            reference_folder: selectedFolder
          });

          if (response.data.error) {
            setError(response.data.error);
          } else {
            setResults(response.data);
          }
        } catch (err) {
          console.error('Error details:', err.response?.data || err.message);
          setError(err.response?.data?.error || 'Error processing image. Please try again.');
        } finally {
          setLoading(false);
        }
      };
    } catch (err) {
      console.error('Error reading file:', err);
      setError('Error reading image file');
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Face Recognition System
        </Typography>

        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Box sx={{ mb: 2 }}>
              <input
                accept="image/*"
                style={{ display: 'none' }}
                id="image-upload"
                type="file"
                onChange={handleImageSelect}
              />
              <label htmlFor="image-upload">
                <Button
                  variant="contained"
                  component="span"
                  startIcon={<CloudUploadIcon />}
                >
                  Select Image
                </Button>
              </label>
            </Box>

            <Box sx={{ mb: 3 }}>
              <FormControl fullWidth variant="outlined" sx={{ mb: 2 }}>
                <InputLabel id="reference-folder-label">Reference Face Set</InputLabel>
                <Select
                  labelId="reference-folder-label"
                  id="reference-folder"
                  value={selectedFolder}
                  onChange={handleFolderChange}
                  label="Reference Face Set"
                  disabled={referenceFolders.length === 0}
                >
                  {referenceFolders.map((folder) => (
                    <MenuItem key={folder} value={folder}>
                      {folder}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>

            {previewUrl && (
              <Box sx={{ mb: 2 }}>
                <img
                  src={previewUrl}
                  alt="Preview"
                  style={{ maxWidth: '100%', maxHeight: '300px' }}
                />
              </Box>
            )}

            <Grid container spacing={2}>
              <Grid item>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleUpload}
                  disabled={!selectedImage || loading}
                >
                  Upload Image
                </Button>
              </Grid>
              <Grid item>
                <Button
                  variant="contained"
                  color="secondary"
                  onClick={handleProcess}
                  disabled={!selectedImage || loading}
                >
                  Process Image
                </Button>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
            <CircularProgress />
          </Box>
        )}

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {results && (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Results
              </Typography>
              {results?.best_similarity_match?.image_base64 && (
                <Box sx={{ mb: 2, textAlign: 'center' }}>
                  <Typography variant="subtitle1" color="primary">
                    Face with Highest Similarity
                  </Typography>
                  <img
                    src={results.best_similarity_match.image_base64}
                    alt="Highest Similarity"
                    style={{ maxWidth: '200px', border: '3px solid #1976d2', borderRadius: 8 }}
                  />
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Similarity: {results.best_similarity_match.similarity.toFixed(3)}
                  </Typography>
                </Box>
              )}
              {results?.reference_face_base64 && (
                <Box sx={{ mb: 2, textAlign: 'center' }}>
                  <Typography variant="subtitle1" color="error">
                    Reference Face Used for Similarity ({results.reference_folder})
                  </Typography>
                  <img
                    src={results.reference_face_base64}
                    alt="Reference Face"
                    style={{ maxWidth: '200px', border: '2px solid #d32f2f', borderRadius: 8 }}
                  />
                </Box>
              )}
              <pre style={{ whiteSpace: 'pre-wrap' }}>
                {JSON.stringify(results, null, 2)}
              </pre>
            </CardContent>
          </Card>
        )}
      </Box>
    </Container>
  );
};

export default Home;
