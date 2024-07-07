FROM minizinc/minizinc:latest

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir /app/results
COPY execution.sh /app/execution.sh
RUN chmod +x /app/execution.sh

ENTRYPOINT ["/app/execution.sh"]
CMD ["cp"]