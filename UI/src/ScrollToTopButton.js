import React, { useState, useEffect } from 'react';
import IconButton from '@mui/material/IconButton';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';

const ScrollToTop = () => {
  const [isVisible, setIsVisible] = useState(false);

  const handleScroll = () => {
    const scrollTop = document.documentElement.scrollTop;
    setIsVisible(scrollTop > 0);
  };

  const handleOnClick = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  useEffect(() => {
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <IconButton
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        display: isVisible ? 'flex' : 'none',
        backgroundColor: '#474747',
        borderRadius: '50%',
        width: '40px',
        height: '40px',
        boxShadow: '0px 4px 4px rgba(0, 0, 0, 0.25)',
      }}
      onClick={handleOnClick}
    >
      <KeyboardArrowUpIcon />
    </IconButton>
  );
};

export default ScrollToTop;
