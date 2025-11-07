import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Form,
  Button,
  Alert,
  Spinner,
  Row,
  Col,
  Card,
  Modal,
  Navbar,
  Container,
  Nav,
  NavDropdown,
} from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import "./BookLayout.css";

const App = () => {
  const [cart, setCart] = useState([]);
  const [showCart, setShowCart] = useState(false);
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState("");
  const [genres, setGenres] = useState([]);
  const [selectedGenre, setSelectedGenre] = useState("");
  const [featured, setFeatured] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState({
    categories: false,
    genres: false,
    recommendations: false,
    search: false,
  });
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const navigate = useNavigate();
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    localStorage.removeItem("userId");

    navigate("/auth");
  };

  useEffect(() => {
    const savedCart = localStorage.getItem("bookCart");
    if (savedCart) setCart(JSON.parse(savedCart));
  }, []);

  useEffect(() => {
    localStorage.setItem("bookCart", JSON.stringify(cart));
  }, [cart]);

  const API = "http://localhost:5000";

  const addDummyPrice = (book) => ({
    ...book,
    price: Math.floor(Math.random() * 800) + 200,
  });

  const cleanBook = (b) => ({
    ...addDummyPrice(b),
    average_rating: Number(b.average_rating) || 0,
    num_pages: Number(b.num_pages) || 0,
    published_year: Number(b.published_year) || 0,
    thumbnail: b.thumbnail || "https://placehold.co/300x200?text=No+Image",
    title: b.title || "Unknown Title",
    authors: b.authors || "Unknown Author",
  });

  // Fetch categories
  useEffect(() => {
    setLoading((prev) => ({ ...prev, categories: true }));
    setError(null);
    fetch(`${API}/categories`)
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) setCategories(data);
        else throw new Error("Invalid categories format");
      })
      .catch((err) => {
        console.error("Category fetch error:", err);
        setError("Failed to load categories");
      })
      .finally(() => setLoading((prev) => ({ ...prev, categories: false })));
  }, []);

  useEffect(() => {
    if (!selectedCategory) {
      setGenres([]);
      return;
    }

    setLoading((prev) => ({ ...prev, genres: true }));
    setError(null);
    fetch(`${API}/genres?category=${encodeURIComponent(selectedCategory)}`)
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data?.genres)) setGenres(data.genres);
        else throw new Error("Invalid genres format");
      })
      .catch((err) => {
        console.error("Genre fetch error:", err);
        setError("Failed to load genres");
      })
      .finally(() => setLoading((prev) => ({ ...prev, genres: false })));
  }, [selectedCategory]);

  const fetchRecommendations = () => {
    if (!selectedCategory || !selectedGenre) return;

    setLoading((prev) => ({ ...prev, recommendations: true }));
    setError(null);
    setSearchResults([]);
    fetch(`${API}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        category: selectedCategory,
        genre: selectedGenre,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        setFeatured(data.featured?.map(cleanBook) || []);
        setRecommendations(data.recommendations?.map(cleanBook) || []);
      })
      .catch((err) => {
        console.error("Recommendation error:", err);
        setError("Failed to load recommendations");
      })
      .finally(() =>
        setLoading((prev) => ({ ...prev, recommendations: false }))
      );
  };

  const searchBooks = (e) => {
    e.preventDefault();
    if (!searchTerm.trim()) return;

    setLoading((prev) => ({ ...prev, search: true }));
    setError(null);
    setFeatured([]);
    setRecommendations([]);
    setSelectedCategory("");
    setSelectedGenre("");

    fetch(`${API}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: searchTerm }),
    })
      .then((res) => res.json())
      .then((data) => {
        setSearchResults(data.results?.map(cleanBook) || []);
      })
      .catch((err) => {
        console.error("Search error:", err);
        setError("Search failed");
      })
      .finally(() => setLoading((prev) => ({ ...prev, search: false })));
  };
  const addToCart = (book) => {
    const newCart = [...cart, cleanBook(book)];
    setCart(newCart);
    alert(`${book.title} has been added to your cart!`);
  };

  const removeFromCart = (index) =>
    setCart((prev) => prev.filter((_, i) => i !== index));
  const clearCart = () => {
    if (window.confirm("Clear your cart?")) setCart([]);
  };
  const totalPrice = cart.reduce((sum, b) => sum + b.price, 0);

  // Checkout function
  const handleCheckout = () => {
    if (cart.length === 0) return;
    const query = cart.map((b) => `${b.title}`).join(" , ");
    window.open(
      `https://www.amazon.in/s?k=${encodeURIComponent(query)}`,
      "_blank"
    );
    setShowCart(false);
  };

  // Render book cards
  const renderBooks = (books, cols = 3, isFeatured = false) => (
    <Row>
      {books.map((b, i) => (
        <Col key={i} md={cols} className="mb-4">
          <Card>
            <Card.Img
              variant="top"
              src={b.thumbnail}
              style={{
                height: cols === 4 ? "200px" : "300px",
                objectFit: "cover",
              }}
              onError={(e) => {
                e.target.src = "https://placehold.co/300x200?text=No+Image";
              }}
            />
            <Card.Body>
              <Card.Title>{b.title}</Card.Title>
              <Card.Text className="text-muted">
                <b>Author: </b> {b.authors}
              </Card.Text>
              {isFeatured && (
                <Card.Text className="text-muted">
                  <b>Rating:</b> {b.average_rating} | {b.num_pages} pages
                </Card.Text>
              )}
              <Card.Text>₹{b.price}</Card.Text>
              <Button onClick={() => addToCart(b)}>Add to Cart</Button>
            </Card.Body>
          </Card>
        </Col>
      ))}
    </Row>
  );

  return (
    <>
      <Navbar bg="dark" variant="dark" expand="md" sticky="top">
        <Container>
          <Navbar.Brand>
            <b>Book Nest</b>
          </Navbar.Brand>

          <Navbar.Toggle aria-controls="navbar-nav" />

          <Navbar.Collapse id="navbar-nav">
            <Nav className="ms-auto">
              <Button
                variant="outline-light"
                className="me-2 mt-2 mt-md-0"
                onClick={() => setShowCart(true)}
                disabled={cart.length === 0}
              >
                Cart ({cart.length}) - ₹{totalPrice}
              </Button>

              <Button
                variant="danger"
                className="mt-2 mt-md-0"
                onClick={handleLogout}
              >
                Logout
              </Button>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      <Container fluid className="mt-4">
        <Row>
          <Col
            md={2}
            className="border-end bg-light d-none d-md-block position-fixed vh-100 mt-1"
            style={{ top: "56px", left: 0, overflowY: "auto" }}
          >
            <h5>Categories</h5>
            {loading.categories ? (
              <Spinner animation="border" size="sm" />
            ) : categories.length > 0 ? (
              categories.map((cat) => (
                <Button
                  key={cat}
                  variant={
                    cat === selectedCategory ? "primary" : "outline-primary"
                  }
                  className="d-block mb-2 w-100 text-start"
                  onClick={() => {
                    setSelectedCategory(cat);
                    setSelectedGenre("");
                    setFeatured([]);
                    setRecommendations([]);
                    setSearchResults([]);
                  }}
                >
                  {cat}
                </Button>
              ))
            ) : (
              <Alert variant="warning">No categories</Alert>
            )}

            {selectedCategory && (
              <>
                <h5 className="mt-4">Genres</h5>
                {loading.genres ? (
                  <Spinner animation="border" size="sm" />
                ) : genres.length > 0 ? (
                  genres.map((g) => (
                    <Button
                      key={g}
                      variant={
                        g === selectedGenre ? "primary" : "outline-primary"
                      }
                      className="d-block mb-2 w-100 text-start"
                      onClick={() => {
                        setSelectedGenre(g);
                        setSearchResults([]);
                      }}
                    >
                      {g}
                    </Button>
                  ))
                ) : (
                  <Alert variant="warning">No genres</Alert>
                )}

                {selectedGenre && (
                  <Button
                    variant="success"
                    className="mt-3 w-100 mb-4"
                    onClick={fetchRecommendations}
                    disabled={loading.recommendations}
                  >
                    {loading.recommendations ? (
                      <Spinner animation="border" size="sm" />
                    ) : (
                      "Show Recommendations"
                    )}
                  </Button>
                )}
              </>
            )}
          </Col>

          <Col xs={12} className="d-md-none mb-4">
            <h5>Categories</h5>
            {loading.categories ? (
              <Spinner animation="border" size="sm" />
            ) : categories.length > 0 ? (
              categories.map((cat) => (
                <Button
                  key={cat}
                  variant={
                    cat === selectedCategory ? "primary" : "outline-primary"
                  }
                  className="d-block mb-2 w-100 text-start"
                  onClick={() => {
                    setSelectedCategory(cat);
                    setSelectedGenre("");
                    setFeatured([]);
                    setRecommendations([]);
                    setSearchResults([]);
                  }}
                >
                  {cat}
                </Button>
              ))
            ) : (
              <Alert variant="warning">No categories</Alert>
            )}

            {selectedCategory && (
              <>
                <h5 className="mt-4">Genres</h5>
                {loading.genres ? (
                  <Spinner animation="border" size="sm" />
                ) : genres.length > 0 ? (
                  genres.map((g) => (
                    <Button
                      key={g}
                      variant={
                        g === selectedGenre ? "primary" : "outline-primary"
                      }
                      className="d-block mb-2 w-100 text-start"
                      onClick={() => {
                        setSelectedGenre(g);
                        setSearchResults([]);
                      }}
                    >
                      {g}
                    </Button>
                  ))
                ) : (
                  <Alert variant="warning">No genres</Alert>
                )}

                {selectedGenre && (
                  <Button
                    variant="success"
                    className="mt-3 w-100 mb-4"
                    onClick={fetchRecommendations}
                    disabled={loading.recommendations}
                  >
                    {loading.recommendations ? (
                      <Spinner animation="border" size="sm" />
                    ) : (
                      "Show Recommendations"
                    )}
                  </Button>
                )}
              </>
            )}
          </Col>

          <Col md={10} className="main-content px-3">
            {error && <Alert variant="danger">{error}</Alert>}

            <Form onSubmit={searchBooks} className="mb-4 d-flex">
              <Form.Control
                type="text"
                placeholder="Search books..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="me-2"
              />
              <Button type="submit" disabled={loading.search}>
                {loading.search ? (
                  <Spinner animation="border" size="sm" />
                ) : (
                  "Search"
                )}
              </Button>
            </Form>

            {searchResults.length > 0 ? (
              <>
                <h4>Search Results</h4>
                {renderBooks(searchResults, 4)}
              </>
            ) : (
              <>
                {featured.length > 0 && (
                  <>
                    <h4>Featured Books</h4>
                    {renderBooks(featured, 4, true)}
                  </>
                )}
                {recommendations.length > 0 && (
                  <>
                    <h4>Recommendations</h4>
                    {renderBooks(recommendations, 3)}
                  </>
                )}
                {!featured.length && !recommendations.length && (
                  <div className="text-center py-5">
                    <h4>Welcome to Book Nest</h4>
                    <p className="text-muted">
                      {selectedCategory
                        ? selectedGenre
                          ? "Click 'Show Recommendations' or search above"
                          : "Select a genre"
                        : "Select a category or search for books"}
                    </p>
                  </div>
                )}
              </>
            )}
          </Col>
        </Row>
      </Container>

      <Modal show={showCart} onHide={() => setShowCart(false)} size="lg">
        <Modal.Header closeButton>
          <Modal.Title>Your Cart ({cart.length} items)</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {cart.length === 0 ? (
            <Alert variant="info">Your cart is empty</Alert>
          ) : (
            <>
              {cart.map((b, i) => (
                <div
                  key={i}
                  className="d-flex justify-content-between align-items-center mb-3 p-2 border-bottom"
                >
                  <div className="d-flex align-items-center">
                    <img
                      src={b.thumbnail}
                      alt={b.title}
                      width="50"
                      height="75"
                      className="me-3"
                      style={{ objectFit: "cover" }}
                      onError={(e) => {
                        e.target.src =
                          "https://placehold.co/50x75?text=No+Image";
                      }}
                    />
                    <div>
                      <h6 className="mb-0">{b.title}</h6>
                      <small className="text-muted">by {b.authors}</small>
                    </div>
                  </div>
                  <div className="d-flex align-items-center">
                    <span className="me-3">₹{b.price}</span>
                    <Button
                      variant="outline-danger"
                      size="sm"
                      onClick={() => removeFromCart(i)}
                    >
                      ×
                    </Button>
                  </div>
                </div>
              ))}
              <div className="d-flex justify-content-between mt-3 p-2 bg-light rounded">
                <h5 className="mb-0">Total:</h5>
                <h5 className="mb-0">₹{totalPrice}</h5>
              </div>
            </>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button
            variant="danger"
            onClick={clearCart}
            disabled={cart.length === 0}
          >
            Clear Cart
          </Button>
          <Button variant="secondary" onClick={() => setShowCart(false)}>
            Continue Shopping
          </Button>
          <Button
            variant="primary"
            onClick={handleCheckout}
            disabled={cart.length === 0}
          >
            Search on Amazon
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
};

export default App;
