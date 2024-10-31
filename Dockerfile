FROM ruby:3.1-alpine

# Install necessary packages
RUN apk update && apk add --no-cache \
    build-base \
    libxml2-dev \
    libxslt-dev \
    nodejs \
    yarn

# Set the working directory
WORKDIR /srv/jekyll

# Copy only the Gemfile into the container
COPY Gemfile /srv/jekyll/

# Install Bundler
RUN gem install bundler

# Create a separate directory for dependencies
RUN mkdir /srv/dependencies

# Set the working directory for bundle install
WORKDIR /srv/dependencies

# Install Jekyll and GitHub Pages dependencies
RUN bundle install --gemfile=/srv/jekyll/Gemfile

# Return to the Jekyll working directory
WORKDIR /srv/jekyll

# Copy the rest of the Jekyll site files into the container
COPY . /srv/jekyll

# Expose the port that Jekyll will serve on
EXPOSE 4000

# Command to run Jekyll server
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0"]


# docker build -t site .
# docker run -p 4000:4000 -v $(pwd):/srv/jekyll site