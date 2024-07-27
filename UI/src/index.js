import React from "react";
import { render } from 'react-dom';
import { ChakraProvider } from "@chakra-ui/react";

import Header from "./components/Header";
import FileUpload from "./components/FileUpload";

function App() {
  return (
    <ChakraProvider>
      <Header />
      <FileUpload />
    </ChakraProvider>
  )
}

const rootElement = document.getElementById("root")
render(<App />, rootElement)