`timescale 1ns/1ps

// ============================================================
// 802.11b 7-bit LFSR Scrambler
// Polynomial: x^7 + x^4 + 1
// ============================================================

module lfsr_scrambler (
    input  wire clk,
    input  wire rst_n,
    input  wire bit_in,
    output reg  bit_out
);

  // 7-bit shift register state
  reg [6:0] state;

  wire fb;  // feedback bit

  // taps: bit3 and bit6 (0-indexed) => corresponds to x^4 and x^7
  assign fb = state[3] ^ state[6];

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state   <= 7'b1111111;   // seed = 0x7F
      bit_out <= 1'b0;
    end else begin
      // scramble
      bit_out <= bit_in ^ fb;

      // update LFSR state
      state <= {state[5:0], fb};
    end
  end

endmodule
