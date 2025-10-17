"""
Image Viewer Component
Display images with annotations for CV tasks
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import io
import base64


class ImageViewer:
    """Component for displaying images with annotations."""
    
    # Color palette for visualizations
    COLORS = [
        "#FF3621", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
        "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896"
    ]
    
    @staticmethod
    def display_image(
        image: Image.Image,
        caption: Optional[str] = None,
        use_column_width: bool = True
    ):
        """
        Display a single image.
        
        Args:
            image: PIL Image
            caption: Optional caption
            use_column_width: Whether to use column width
        """
        st.image(image, caption=caption, use_column_width=use_column_width)
    
    @staticmethod
    def display_image_grid(
        images: List[Image.Image],
        captions: Optional[List[str]] = None,
        columns: int = 3
    ):
        """
        Display images in a grid layout.
        
        Args:
            images: List of PIL Images
            captions: Optional list of captions
            columns: Number of columns in grid
        """
        if not images:
            st.warning("No images to display")
            return
        
        cols = st.columns(columns)
        for idx, image in enumerate(images):
            col_idx = idx % columns
            with cols[col_idx]:
                caption = captions[idx] if captions and idx < len(captions) else None
                st.image(image, caption=caption, use_column_width=True)
    
    @staticmethod
    def draw_bounding_boxes(
        image: Image.Image,
        boxes: List[List[float]],
        labels: Optional[List[str]] = None,
        scores: Optional[List[float]] = None,
        format: str = "xyxy"
    ) -> Image.Image:
        """
        Draw bounding boxes on an image.
        
        Args:
            image: PIL Image
            boxes: List of bounding boxes
            labels: Optional list of labels
            scores: Optional list of confidence scores
            format: Box format ("xyxy" or "xywh")
            
        Returns:
            Image with drawn boxes
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
        
        for idx, box in enumerate(boxes):
            color = ImageViewer.COLORS[idx % len(ImageViewer.COLORS)]
            
            # Convert box format if needed
            if format == "xywh":
                x, y, w, h = box
                x1, y1, x2, y2 = x, y, x + w, y + h
            else:
                x1, y1, x2, y2 = box
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            if labels or scores:
                label_text = ""
                if labels and idx < len(labels):
                    label_text = labels[idx]
                if scores and idx < len(scores):
                    label_text += f" {scores[idx]:.2f}"
                
                # Draw label background
                text_bbox = draw.textbbox((x1, y1), label_text, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((x1, y1), label_text, fill="white", font=font)
        
        return img
    
    @staticmethod
    def draw_segmentation_mask(
        image: Image.Image,
        mask: np.ndarray,
        alpha: float = 0.5,
        color_map: Optional[Dict[int, Tuple[int, int, int]]] = None
    ) -> Image.Image:
        """
        Overlay a segmentation mask on an image.
        
        Args:
            image: PIL Image
            mask: Numpy array with class IDs or binary mask
            alpha: Transparency of overlay
            color_map: Optional mapping of class IDs to RGB colors
            
        Returns:
            Image with mask overlay
        """
        img = image.copy().convert("RGBA")
        
        # Create colored mask
        h, w = mask.shape[:2]
        colored_mask = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Get unique classes
        unique_classes = np.unique(mask)
        
        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue
            
            # Get color for this class
            if color_map and class_id in color_map:
                color = color_map[class_id]
            else:
                color_idx = class_id % len(ImageViewer.COLORS)
                color_hex = ImageViewer.COLORS[color_idx]
                color = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
            
            # Set pixels for this class
            class_mask = mask == class_id
            colored_mask[class_mask] = (*color, int(255 * alpha))
        
        # Convert to PIL and composite
        mask_img = Image.fromarray(colored_mask, mode="RGBA")
        mask_img = mask_img.resize(img.size, Image.LANCZOS)
        
        result = Image.alpha_composite(img, mask_img)
        return result.convert("RGB")
    
    @staticmethod
    def create_comparison_view(
        image1: Image.Image,
        image2: Image.Image,
        label1: str = "Original",
        label2: str = "Prediction"
    ):
        """
        Display two images side by side for comparison.
        
        Args:
            image1: First image
            image2: Second image
            label1: Label for first image
            label2: Label for second image
        """
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{label1}**")
            st.image(image1, use_column_width=True)
        with col2:
            st.markdown(f"**{label2}**")
            st.image(image2, use_column_width=True)
    
    @staticmethod
    def get_image_info(image: Image.Image) -> Dict[str, Any]:
        """
        Get information about an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with image information
        """
        return {
            "size": image.size,
            "mode": image.mode,
            "format": image.format,
            "width": image.width,
            "height": image.height,
        }
    
    @staticmethod
    def display_image_info(image: Image.Image):
        """
        Display image information in a formatted way.
        
        Args:
            image: PIL Image
        """
        info = ImageViewer.get_image_info(image)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Width", f"{info['width']}px")
        with col2:
            st.metric("Height", f"{info['height']}px")
        with col3:
            st.metric("Mode", info['mode'])
        with col4:
            st.metric("Format", info.get('format', 'N/A'))
    
    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image
            
        Returns:
            Base64 encoded string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    @staticmethod
    def download_button(image: Image.Image, filename: str = "image.png", label: str = "Download Image"):
        """
        Create a download button for an image.
        
        Args:
            image: PIL Image
            filename: Filename for download
            label: Button label
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        st.download_button(
            label=label,
            data=img_bytes,
            file_name=filename,
            mime="image/png"
        )
    
    @staticmethod
    def annotate_classification(
        image: Image.Image,
        predicted_class: str,
        confidence: float,
        true_class: Optional[str] = None
    ) -> Image.Image:
        """
        Annotate image with classification results.
        
        Args:
            image: PIL Image
            predicted_class: Predicted class name
            confidence: Prediction confidence
            true_class: Optional ground truth class
            
        Returns:
            Annotated image
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
        except:
            font = ImageFont.load_default()
        
        # Create annotation text
        text = f"Predicted: {predicted_class} ({confidence:.2%})"
        if true_class:
            text += f"\nGround Truth: {true_class}"
            color = "#2ca02c" if predicted_class == true_class else "#d62728"
        else:
            color = "#FF3621"
        
        # Draw text with background
        text_bbox = draw.textbbox((10, 10), text, font=font)
        padding = 10
        draw.rectangle(
            [text_bbox[0] - padding, text_bbox[1] - padding,
             text_bbox[2] + padding, text_bbox[3] + padding],
            fill=color
        )
        draw.text((10, 10), text, fill="white", font=font)
        
        return img

