// module.register is deprecated from Node 26 (DEP0205); registerHooks exists from 22.15 / 23.5.
import { register, registerHooks } from 'node:module';
import { resolve } from './alias-hooks.js';

if (typeof registerHooks === 'function') registerHooks({ resolve });
else register('./alias-hooks.js', import.meta.url);
