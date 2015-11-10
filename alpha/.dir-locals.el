;; Generate the first build files using:
;;;;    cd build/ && cmake -GNinja -D CMAKE_BUILD_TYPE=Release ../ && ninja
;;;; or cd debug/ && cmake -GNinja -D CMAKE_BUILD_TYPE=Debug ../ && ninja
;; and then use the much simpler commands, who don't need all these arguments

((nil .  ((projectile-project-compilation-cmd . "ninja -C build")
          (projectile-project-run-cmd . "build/alpha/inference")
          )
      )
 )
