// Text formatters the model frames share with the page test.
import { formatCount } from '@math/core.js';

// gpt-oss's size as decoder-anatomy prints it (P4-R6 / D8): "116.83B" total, "5.13B" active.
export const COUNT_FORMAT = Object.freeze({ total: (n) => formatCount(n, { digits: 5 }), active: (n) => formatCount(n) });
