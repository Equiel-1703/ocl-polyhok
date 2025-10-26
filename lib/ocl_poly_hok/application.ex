defmodule OCLPolyHok.Application do
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      %{
        id: :debug_logs_agent,
        start: {Agent, :start_link, [fn -> false end, [name: :debug_logs_agent]]}
      }
    ]

    opts = [strategy: :one_for_one, name: OCLPolyHok.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
