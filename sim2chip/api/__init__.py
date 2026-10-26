"""The web API's routes, one module per part of the app. server.create_app makes the shared state (the library, the
job service, who is asking and what they may start) and passes it to each module's add(app, ctx)."""
