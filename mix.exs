defmodule OCLPolyHok.MixProject do
  use Mix.Project

  def project do
    [
      app: :ocl_poly_hok,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {OCLPolyHok.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.9.2"},
      {:exla, "~> 0.9"}
    ]
  end
end
