.PHONY: serve build clean install update

# Default target
all: install serve

# Install dependencies
install:
	bundle install

# Serve the site locally
serve:
	bundle exec jekyll serve

# Build the site
build:
	bundle exec jekyll build

# Clean the site
clean:
	bundle exec jekyll clean
	rm -rf _site
	rm -rf .jekyll-cache

# Update dependencies
update:
	bundle update

# Build and serve with drafts
drafts:
	bundle exec jekyll serve --drafts

# Build and serve with future posts
future:
	bundle exec jekyll serve --future

# Build and serve with both drafts and future posts
all-posts:
	bundle exec jekyll serve --drafts --future

# Help command
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make serve      - Serve the site locally"
	@echo "  make build      - Build the site"
	@echo "  make clean      - Clean the site"
	@echo "  make update     - Update dependencies"
	@echo "  make drafts     - Serve with draft posts"
	@echo "  make future     - Serve with future posts"
	@echo "  make all-posts  - Serve with both drafts and future posts" 